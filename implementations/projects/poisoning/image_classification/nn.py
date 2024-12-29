from __future__ import annotations
from copy import deepcopy
import dataclasses
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.optim import Optimizer, AdamW
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Metric, Accuracy

from .datasets import class_weights
from .utils import tqdm, trange


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: _Loss,
        optimizer: Optimizer,
        keep_pbars=True,
        metric: Metric = None,
    ):
    #optimizer.zero_grad()
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    # Make a nice progress bar
    pbar = tqdm(desc='Train loop', total=len(dataloader.dataset), leave=keep_pbars)
    for X, y in dataloader:
        # Compute prediction and loss
        logits = model(X)
        loss = loss_fn(logits, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.n += len(X)
        pbar.set_postfix(loss=loss.item())
    
    if keep_pbars:
        # Fixes a bug where the HTML output does not survive after closing notebook
        print(pbar)
    pbar.close()

def test_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: _Loss,
        keep_pbars=False,
        metric: Metric = None,
    ):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    test_loss = 0.0
    # Make a nice progress bar
    pbar = tqdm(desc='Test loop', total=len(dataloader.dataset), leave=keep_pbars)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            logits = model(X)
            loss = loss_fn(logits, y).item()
            test_loss += loss

            metrics = {}
            if metric is not None:
                metric_name = metric._get_name()
                metric_on_batch = metric(logits, y).item()
                metrics[metric_name] = metric_on_batch

            pbar.n += len(X)
            pbar.set_postfix(
                **metrics,
                avg_loss=(test_loss / (i + 1)),
            )

    if keep_pbars:
        # Fixes a bug where the HTML output does not survive after closing notebook
        print(pbar)
    pbar.close()

def train_loop(
        model: nn.Module,
        train_dataloader: DataLoader,
        loss_fn: _Loss,
        optimizer: Optimizer,
        epochs: int,
        keep_pbars=True,
        metric: Metric = None,
    ):
    """
    Run the training loop on the model without testing.
    """
    for _ in trange(epochs, desc='Train epochs', unit='epoch', leave=keep_pbars):
        train_epoch(
            model,
            train_dataloader,
            loss_fn, optimizer,
            keep_pbars=keep_pbars,
            metric=metric,
        )

def train_test_loop(
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        loss_fn: _Loss,
        optimizer: Optimizer,
        epochs: int,
        keep_pbars=True,
        metric: Metric = None,
    ):
    """
    Run the training and testing loop on the model.

    If `test_dataloader` is `None`, no testing is performed after training.
    """
    for _ in trange(epochs, desc='Train epochs', unit='epoch', leave=keep_pbars):
        train_epoch(model, train_dataloader, loss_fn, optimizer, keep_pbars=keep_pbars)
        test_epoch(model, test_dataloader, loss_fn, keep_pbars=keep_pbars, metric=metric)


def distillation_epoch(
        teacher: nn.Module,
        student: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer,
        criterion,
        keep_pbars=True,
    ):
    """
    Perform a single epoch of knowledge distillation in a student-teacher method.

    # Arguments
    - `teacher` : a pretrained, reference model, usually the larger model.
    - `student` : a possibly untrained model, usually the smaller one.
    - `dataloader` : the data to compare the models on
    - `criterion` : a minimizer that takes three arguments:
        - `logits_student` : the student model predictions
        - `logits_teacher` : the teacher model predictions
        - `target` : the target in the dataset
    """
    student.train()
    teacher.eval()

    pbar = tqdm(
        desc='Transfer learning epoch',
        total=len(dataloader.dataset),
        leave=keep_pbars,
    )

    for X, target in dataloader:
        # Compute teacher and student predictions
        logits_teacher = teacher(X)
        logits_student = student(X)

        loss = criterion(logits_student, logits_teacher, target)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.n += len(X)
        pbar.set_postfix(loss=loss.item())

    if keep_pbars:
        # Fixes a bug where the HTML output does not survive after closing notebook
        print(pbar)
    pbar.close()



@dataclass(frozen=True)
class OptimizerParams:
    # Learning rate for optimizer.
    lr: float = 1e-3

    # Regularization coefficient for the optimizer.
    weight_decay: float = 1e-3

@dataclass(frozen=True)
class Hyperparameters:
    """
    A reusable set of hyperparameters and training settings.

    # Usage example

    ```python
    hp = Hyperparameters()
    hp.optimizer_params.lr = 1e-4

    train_test_loop(
        model, train_loader, test_loader,
        hp.train_test_params(model),
    )
    ```
    """

    # Note that the performance can decrease with a larger batch size.
    batch_size: int = 64

    # Since test time requires less memory than training, we can use a larger batch size.
    inference_batch_size: int = 128

    # Make training time as short as possible.
    # This is fine since the dataset is large and features are easy to learn.
    epochs: int = 1

    loss_fn: _Loss = CrossEntropyLoss()

    # We use AdamW, which is one of the most popular optimizers in deep learning.
    # It converges faster than SGD and is more efficient.
    # However, its performance is also much more sensitive to hyperparameters and noise in the dataset.
    optimizer: type = AdamW

    optimizer_params: OptimizerParams = OptimizerParams()

    metric: Metric = Accuracy(task='multiclass', num_classes=10)

    def make_optimizer(self, model: nn.Module) -> Optimizer:
        return self.optimizer(
            model.parameters(),
            **dataclasses.asdict(self.optimizer_params),
        )

    def train_loop_params(self, model: nn.Module) -> dict:
        return dict(
            loss_fn=self.loss_fn,
            optimizer=self.make_optimizer(model),
        )
    
    def test_loop_params(self) -> dict:
        return dict(loss_fn=self.loss_fn, metric=self.metric)
    
    def train_test_params(
            self,
            model: nn.Module,
            dataset: Dataset = None,
            tune_loss_fn=False,
            #tune_optimizer=False,
            #tune_epochs=False,
            #tune_batch_size=False,
        ) -> dict:
        """
        A utility function that returns appropriate keyword hyperparameters
        for training a model with `train_test_loop`.
        
        The hyperparameters can be adjusted according to the keyword arguments
        to this function.
        
        In particular, the loss function can be modified to handle class imbalance
        in the dataset.

        TODO: perform hyperparameter tuning here
        """
        loss_fn = deepcopy(self.loss_fn)
        if tune_loss_fn and isinstance(loss_fn, _WeightedLoss):
            set_loss_weights(loss_fn, dataset)
        
        return dict(
            epochs=self.epochs,
            loss_fn=loss_fn,
            optimizer=self.make_optimizer(model),
            metric=self.metric,
        )


def set_loss_weights(loss: _WeightedLoss, dataset: Dataset):
    """
    A utility function to set the loss weights in place, according to
    the class weights of a dataset.
    """
    _, counts = class_weights(dataset)
    loss.weight = counts.min() / torch.tensor(counts)