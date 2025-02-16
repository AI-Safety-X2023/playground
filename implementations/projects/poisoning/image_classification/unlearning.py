from __future__ import annotations

from contextlib import contextmanager
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer, SGD

from .utils import tqdm, trange
from .nn import MetricLogger, distillation_epoch, train_val_loop


def model_layers(model: nn.Module, first: int, last: int):
    """
    Iterate on model layers from `first` to `last` (inclusive).

    `first` and `last` may be negative, for instance `-1` refers to the last layer.
    """
    num_layers = sum(1 for _ in model.modules()) 
    for (i, layer) in enumerate(model.modules()):
        if num_layers - first <= i <= num_layers - last:
            yield layer

def forget_last_layers(model: nn.Module, k: int):
    for layer in model_layers(model, -k, -1):
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        else:
            for param in layer.parameters():
                # Reset the parameter with gaussian noise
                std = torch.std(param.data)
                param.data = torch.randn_like(param.data) * std


@contextmanager
def unlearning_last_layers(model: nn.Module, k: int, mode='cfk'):
    """
    Perform machine unlearning by retraining the k last layers.

    # Arguments
    - `k` : the number of layers to retrain
    - `mode` :
        - `'euk'` : use the EUk algorithm, i.e. retrain the last layers form scratch
        - `'cfk'` : use the CFk algorithm, i.e. don't retrain from scratch (default)
    
    # Example

    ```python
    with unlearning_last_layers(model, 3, mode='euk'):
        train_val_loop(model, ...)
    ```
    """
    def _set_layer_grad_updates(layer: nn.Module, grad: bool):
        for param in layer.parameters():
            param.requires_grad = grad

    # Alternative: optimizer = Adam(last_layers_module.parameters())
    def _freeze_layers_except_last(model: nn.Module, k: int):
        for layer in model_layers(model, 0, -(k-1)):
            _set_layer_grad_updates(layer, False)

    def _unfreeze_layers_except_last(model: nn.Module, k: int):
        for layer in model_layers(model, 0, -(k-1)):
            _set_layer_grad_updates(layer, False)
    
    algos = ['cfk', 'euk']
    if mode not in algos:
        raise ValueError(f'Invalid mode: `{mode}`. Valid algorithms: {algos}')

    if mode == 'euk':
        forget_last_layers(model, k)
    
    _freeze_layers_except_last(model, k)
    yield
    _unfreeze_layers_except_last(model, k)


def gradient_descent(
        model: nn.Module,
        retain_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: _Loss,
        optimizer: SGD,
        epochs: int,
        **kwargs,
    ):
    """
    Gradient descent unlearning method.

    Simply perform gradient descent on the retain set (Neel et al.)\\
    https://arxiv.org/abs/2007.02923
    """
    return train_val_loop(
        model,
        retain_loader, val_loader,
        loss_fn, optimizer, epochs,
        early_stopping=False,
        **kwargs,
    )

def gradient_ascent(
        model: nn.Module,
        forget_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: _Loss,
        optimizer: SGD,
        epochs: int,
        **kwargs,
    ):
    """
    Perform gradient ascent on the forget set.

    We use the approximate unlearning method suggested by Jang et al.\\
    https://arxiv.org/abs/2210.01504
    """
    criterion = lambda y, y_true: -loss_fn(y, y_true)
    return train_val_loop(
        model,
        forget_loader, val_loader,
        criterion, optimizer, epochs,
        early_stopping=False,
        **kwargs,
    )

class NoisySGD(SGD):
    """
    Noisy gradient optimizer from Chien et al.

    Reference paper: https://arxiv.org/abs/2401.10371
    """

    def __init__(self, *args, noise_scale=np.sqrt(1e-7), **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: add this hyperparameter in param groups
        self.noise_scale = noise_scale

    @torch.no_grad()
    def step(self, closure=None):
        # Perform normal SGD first
        loss = super().step(closure)

        # Add noise to the model parameters
        for group in self.param_groups:
            for param in group['params']:
                noise = torch.randn_like(param.data) * self.noise_scale
                param.data -= group['lr'] * noise
        
        return loss


# https://github.com/pytorch/pytorch/issues/23900
def cycle(iterable: DataLoader):
    """
    Cycle through the elements of a dataloader infinitely.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def neg_grad_plus(
        model: nn.Module,
        retain: DataLoader,
        forget: DataLoader,
        loss_fn: _Loss,
        optimizer: Optimizer,
        beta=0.999,
        keep_pbars=True,
    ):
    """
    NegGrad+ algorithm from Kurmanji et al. A finetuning-based unlearning algorithm
    that balances gradient ascent on the forget set and gradient descent on the retain set.

    This function performs one training epoch.

    # Parameters
    `beta` : the loss function coefficient on the retain set.
             The coefficient on the forget set is `1 - beta`.

    Reference paper: https://arxiv.org/abs/2302.09880
    """
    model.train()

    logger = MetricLogger(
        desc='NegGrad+', total=len(forget.dataset), keep_pbars=keep_pbars,
    )
    
    # This prevents errors in case `retain` has less many elements than `forget`.
    # Usually not needed.
    retain_iter = cycle(retain)

    # We perform the unlearning loop on all of the forget set.
    for X_f, y_f in forget:
        X_r, y_r = next(retain_iter)

        # Compute prediction and loss
        loss_f = loss_fn(model(X_f), y_f)
        loss_r = loss_fn(model(X_r), y_r)

        loss = beta * loss_r - (1. - beta) * loss_f

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        logger.compute_metrics(X_f, y_f, None, loss.item())
    
    logger.finish()


def oracle_unlearning(
        model: nn.Module,
        forget_uncorrupted: DataLoader,
        forget: DataLoader,
        loss_fn: _Loss,
        optimizer: Optimizer,
        beta=0.9,
        keep_pbars=True,
    ):
    """
    A variation of NegGrad+ where the retain set is replaced by the forget set
    before poisoning.

    This unlearning method requires the defender has access to the original, clean data.
    However, it is still computationally worthwhile compared to exact unlearning.
    """
    model.train()

    logger = MetricLogger(
        desc='OracleUnlearning', total=len(forget.dataset), keep_pbars=keep_pbars,
    )

    # We perform the unlearning loop on all of the forget set.
    for (X_r, y_r), (X_f, y_f) in zip(forget_uncorrupted, forget):

        # Compute prediction and loss
        loss_f = loss_fn(model(X_f), y_f)
        loss_r = loss_fn(model(X_r), y_r)

        loss = beta * loss_r - (1. - beta) * loss_f

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        logger.compute_metrics(X_f, y_f, None, loss.item())
    
    logger.finish()


def scrub_unlearning_epoch(
        teacher: nn.Module,
        student: nn.Module,
        forget: DataLoader,
        optimizer: Optimizer,
        beta=0.01,
        keep_pbars=True,
    ):
    """
    This corresponds to the Algorithm 2: `DO-MAX-EPOCH` in the SCRUB paper.
    """
    def criterion(logits_student, logits_teacher, _target):
        # Note: `logits_student` is the reference distribution.
        # The output is assumed to be unnormalized (positive or negative).
        # FIXME: this requires some output regularization to avoid overflow.
        kl = F.kl_div(logits_student, logits_teacher, log_target=True, reduction='mean')
        return - beta * kl
    
    return distillation_epoch(
        teacher, student,
        forget,
        optimizer,
        criterion,
        keep_pbars,
    )

def scrub_learning_epoch(
        teacher: nn.Module,
        student: nn.Module,
        retain: DataLoader,
        loss_fn: _Loss,
        optimizer: Optimizer,
        alpha=0.1,
        gamma=0.9,
        keep_pbars=True,
    ):
    """
    This corresponds to the Algorithm 3: `DO-MIN-EPOCH` in the SCRUB paper.
    """
    def criterion(logits_student, logits_teacher, target):
        loss = loss_fn(logits_student, target)
        # Note: `logits_student` is the reference distribution.
        # The output is assumed to be unnormalized (positive or negative).
        # FIXME: this requires some output regularization to avoid overflow.
        kl = F.kl_div(logits_student, logits_teacher, log_target=True, reduction='mean')
        return alpha * kl + gamma * loss

    return distillation_epoch(
        teacher, student,
        retain,
        optimizer,
        criterion,
        keep_pbars,
    )

def scrub(
        teacher: nn.Module,
        student: nn.Module,
        retain: DataLoader,
        forget: DataLoader,
        loss_fn: _Loss,
        optimizer: Optimizer,
        max_steps: int,
        steps: int,
        alpha=0.1,
        beta=0.01,
        gamma=0.9,
        keep_pbars=True,
    ):
    """
    SCRUB algorithm from Kurmanji et al. A state-of-the-art unlearning algorithm
    that selectively trains a student model on a teacher model.

    # Parameters
    - `alpha` : KL divergence coefficient on the retain set
    - `gamma` : loss function coefficient on the retain set

    Reference paper: https://arxiv.org/abs/2302.09880

    TODO: implement SCRUB+R
    """
    teacher.eval()
    student.train()

    for epoch in trange(steps, desc='SCRUB epoch', leave=keep_pbars):
        if epoch < max_steps:
            scrub_unlearning_epoch(
                teacher, student,
                forget,
                optimizer,
                beta,
                keep_pbars,
            )
        scrub_learning_epoch(
            teacher, student,
            retain,
            loss_fn, optimizer,
            alpha, gamma,
            keep_pbars,
        )
        

