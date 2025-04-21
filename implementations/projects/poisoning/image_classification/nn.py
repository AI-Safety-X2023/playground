from __future__ import annotations
from copy import deepcopy
import dataclasses
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Metric, MeanMetric, MetricTracker
from torchmetrics.classification import MulticlassAccuracy

from .accel import BEST_DEVICE
from .datasets import class_weights
from .utils import tqdm, trange


class MetricLogger:
    def __init__(
            self, *metrics: Metric,
            device=BEST_DEVICE,
            desc='Train loop', total: int = None, keep_pbars=True
        ):
        self.metrics = {
            metric._get_name(): metric.to(device)
            for metric in metrics if metric is not None
        }
        self.avg_loss = MeanMetric().to(device)

        # Make a nice progress bar
        self.pbar = tqdm(desc=desc, total=total, leave=keep_pbars)
        # Forces storage of description even if progress bar is disabled
        self.pbar.desc = desc or ''

    def compute_metrics(
            self,
            X: Tensor, y: Tensor, logits: Tensor, loss: float,
        ) -> dict[str, Metric]:
        self.avg_loss(loss)
        metric_values = {'avg_loss': self.avg_loss.compute().item()}
        for name, metric in self.metrics.items():
            if isinstance(metric, MetricTracker):
                # Track the metric over each step
                metric.increment()

            value = metric(logits, y)
            
            if isinstance(value, dict):
                for n, v in value.items():
                    assert isinstance(v, Tensor)
                    if v.ndim >= 1:
                        v = v.mean()
                    metric_values[n] = v.item()
            else:
                assert isinstance(value, Tensor)
                if value.ndim >= 1:
                        value = value.mean()
                metric_values[name] = value.item()

        self.pbar.n += len(X)
        self.pbar.set_postfix(**metric_values)
    
    def compute_additional_metrics(self, metric_names: list[str], *args):
        for name in metric_names:
            self.metrics[name](*args)

    def finish(self):            
        if self.pbar.leave:
            # Fixes a bug where the HTML output does not survive after closing notebook
            print(self.pbar)
            if self.pbar.disable:
                # Fixes a bug in tqdm: pbar info is not displayed
                print(f'{self.pbar.desc}\t-', self.pbar.postfix)
        self.pbar.close()

class Logs:
    """An object that tracks training and validation metrics over epochs.
    
    Parameters:
        train_metrics (list of dict of Metric):
            a list of computed metrics indexed by the training epoch.
        val_metrics (list of dict of optional Metric):
            a list of computed metrics indexed by the validation epoch.
            `val_metrics[epoch]` is empty when no validation was performed at `epoch`.
        unlearn_metrics (list of dict of optional Metric):
            a list of computed metrics indexed by the unlearning epoch.
            `unlearn_metrics[epoch]` is empty when no unlearning was performed at `epoch`.
    """
    def __init__(self):
        self.train_metrics: list[dict[str, Metric]] = []
        self.val_metrics: list[dict[str, Metric]] = []
        self.unlearn_metrics:  list[dict[str, Metric]] = []
    
    def update_train_epoch(self, train_logger: MetricLogger):
        """Update the training metrics of an epoch.

        Parameters:
            train_logger (MetricLogger, optional): the logger returned by
                the training epoch.
        """
        self.train_metrics.append(train_logger.metrics)
    
    def update_val_epoch(self, val_logger: MetricLogger = None):
        """Update the validation metrics of an epoch.

        Parameters:
            val_logger (MetricLogger, optional): the logger returned by
                the validation epoch. `None` if no validation performed at this step.
        """
        if val_logger is None:
            metrics = {}
        else:
            metrics = val_logger.metrics
        self.val_metrics.append(metrics)
    
    def update_unlearn_epoch(self, unlearn_logger: MetricLogger = None):
        """Update the unlearn metrics of an epoch.

        Parameters:
            unlearn_logger (MetricLogger, optional): the logger returned by
                the validation epoch. `None` if no validation performed at this step.
        """
        if unlearn_logger is None:
            metrics = {}
        else:
            metrics = unlearn_logger.metrics
        self.unlearn_metrics.append(metrics)


def _detect_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device

def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: _Loss,
        optimizer: Optimizer,
        keep_pbars=True,
        metric: Metric = None,
    ):
    device = _detect_device(model)
    #optimizer.zero_grad()
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()

    logger = MetricLogger(
        metric,
        desc='Train loop', total=len(dataloader.dataset), keep_pbars=keep_pbars,
    )

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        logits = model(X)
        loss = loss_fn(logits, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        logger.compute_metrics(X, y, logits, loss.item())
    
    logger.finish()
    return logger

def test_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: _Loss,
        keep_pbars=False,
        metric: Metric = None,
    ):
    device = _detect_device(model)
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    
    logger = MetricLogger(
        metric,
        desc='Test epoch', total=len(dataloader.dataset), keep_pbars=keep_pbars,
    )

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)    
            logits = model(X)
            loss = loss_fn(logits, y).item()

            logger.compute_metrics(X, y, logits, loss)

    logger.finish()
    return logger

def train_loop(
        model: nn.Module,
        train_dataloader: DataLoader,
        loss_fn: _Loss,
        optimizer: Optimizer,
        epochs: int,
        *,
        lr_scheduler: LRScheduler = None,
        keep_pbars=True,
        metric: Metric = None,
    ) -> Logs:
    """Run the training loop on the model without testing.
    
    Parameters:
        keep_pbars (bool, optional): whether to keep progress bars. Defaults to `True`.
        metric (torchmetrics.Metric, optional): a metric (or a metric collection) that takes
            true labels and logits as arguments.
    
    Returns:
        logs (Logs): the complete history of tracked metrics.
    """
    logs = Logs()
    for epoch in trange(epochs, desc='Train epochs', unit='epoch', leave=keep_pbars):
        logger = train_epoch(
            model,
            train_dataloader,
            loss_fn, optimizer,
            keep_pbars=keep_pbars,
            metric=metric,
        )
        logs.update_train_epoch(logger)
        if lr_scheduler is not None:
            lr_scheduler.step()
    return logs

def train_val_loop(
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        loss_fn: _Loss,
        optimizer: Optimizer,
        epochs: int,
        *,
        lr_scheduler: LRScheduler = None,
        keep_pbars=True,
        metric: Metric = None,
        validate_every: int = 2,
        early_stopping = True,
    ) -> Logs:
    """Run the training loop on the model with periodic validation.

    If `val_dataloader` is `None`, no validation is performed.

    If `early_stopping` is True, the training loop exits when validation loss starts decreasing.
    
    Returns:
        logs (Logs): the training and validation logged metrics at each epoch.
    """
    logs = Logs()
    val_loss = float('inf')
    for epoch in trange(epochs, desc='Train epochs', unit='epoch', leave=keep_pbars):
        logger = train_epoch(
            model, train_dataloader, loss_fn, optimizer,
            keep_pbars=keep_pbars, metric=metric,
        )
        logs.update_train_epoch(logger)

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if val_dataloader is not None and epoch % validate_every == 0:
            logger = test_epoch(
                model, val_dataloader, loss_fn,
                keep_pbars=keep_pbars, metric=metric,
            )
            logs.update_val_epoch(logger)
            next_val_loss = logger.avg_loss.compute()
            if early_stopping and next_val_loss > val_loss:
                print(f"Epoch {epoch}: validation loss stopped improving, exiting train loop.")
                break
            val_loss = next_val_loss
        else:
            logs.update_val_epoch()
    
    return logs


def distillation_epoch(
        teacher: nn.Module,
        student: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer,
        criterion,
        keep_pbars=True,
        metric: Metric = None,
    ) -> MetricLogger:
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

    logger = MetricLogger(
        metric,
        desc='Transfer learning epoch', total=len(dataloader.dataset), keep_pbars=keep_pbars,
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

        logger.compute_metrics(X, target, logits_student, loss)

    logger.finish()
    return logger



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

    train_val_loop(
        model, train_loader, val_loader,
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

    metric: Metric = MulticlassAccuracy(num_classes=10)

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
        for training a model with `train_val_loop`.
        
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


def find_lr(
        model: nn.Module,
        criterion: _Loss,
        optimizer: Optimizer,
        train_loader: DataLoader,
        *,
        init_value = 1e-6,
        final_value=1.,
        beta = 0.98,
    ):
    """
    Find a good learning rate for the optimizer.

    The learning rate is store in the optimizer parameters.
    """
    # https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    num = len(train_loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for X, y in train_loader:
        batch_num += 1
        # Get the loss for this mini-batch of inputs/outputs
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)

        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) *loss.data[0]
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(np.log10(lr))

        # Do the SGD step
        loss.backward()
        optimizer.step()

        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses