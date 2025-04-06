from copy import deepcopy
from enum import Enum
import torch
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss, CrossEntropyLoss
from torch.optim import Optimizer, SGD, Adam, AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.functional import total_variation
import torchinfo
from .utils import trange



class GradientAttack(Enum):
    """
    A type of gradient attack.
    """
    
    ASCENT = 0              # Gradient Ascent (Blanchard et al., 2017)
    ORTHOGONAL = 1          # Orthogonal Gradient
    LITTLE_IS_ENOUGH = 2    # Little is Enough (Shejwalkar & Houmansadr, 2021)



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

class GradientEstimator:
    """A class for average model gradient estimation."""
    def average_clean_gradient(self, model: nn.Module, criterion: _Loss) -> Tensor:
        """
        Estimate the average gradient on a clean-distributed dataset.
        """
        raise NotImplementedError

class OmniscientGradientEstimator(GradientEstimator):
    """
    Estimates the average gradient assuming it has already been computed
    on a mini-batch with loss backpropagation.

    # Example
    ```python
    grad_estim = OmniscientGradientEstimator(batch_size)
    loss = criterion(model(X), y)
    loss.backward()
    avg_clean_gradient = grad_estim.average_clean_gradient(model, criterion)
    ```
    """
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
    
    def average_clean_gradient(self, model: nn.Module, criterion: _Loss) -> Tensor:
        return average_model_gradient(model, self.batch_size)
    

class ShadowGradientEstimator(GradientEstimator):
    """
    Estimate the average clean gradient with an auxiliary dataset
    that is similarly distributed to the training dataset.
    """
    def __init__(self, aux_loader: DataLoader):
        self.aux_loader = cycle(aux_loader)
    
    def average_clean_gradient(self, model: nn.Module, criterion: _Loss) -> Tensor:
        # TODO: estimate on next mini-batch of aux_loader
        raise NotImplementedError



# TODO: gradients per parameter instead?
def model_gradients(model: nn.Module) -> Tensor:
    """
    Returns the model gradients without detaching them.
    """
    grads = [
        param.grad.flatten()
        for param in model.parameters()
    ]
    return torch.cat(grads)

def average_model_gradient(model: nn.Module, batch_size: int) -> Tensor:
    """
    Returns the model gradient averaged over the batch size.

    Assumes the gradients have already been computed with loss backpropagation.
    """
    return model_gradients(model).detach().clone() / batch_size