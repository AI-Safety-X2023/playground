from abc import ABC, abstractmethod
from warnings import deprecated

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
import torch.func as ft

from . import utils

# Inspired by torchjd: https://torchjd.org
class Aggregator(nn.Module, ABC):
    """An abstract class for gradient aggregators.
    
    It aggregates matrices of dimension `(b, d)` into row gradients of dimension `d`.
    """
    @abstractmethod
    def forward(self, matrix: Tensor) -> Tensor:
        """Computes the aggregation from the input matrix."""
    
class Mean(Aggregator):
    """Computes the coordinate-wise mean."""
    def forward(self, matrix: Tensor) -> Tensor:
        return matrix.mean(dim=0)

class Stddev(Aggregator):
    """Computes the coordinate-wise standard deviation."""
    def forward(self, matrix: Tensor) -> Tensor:
        raise NotImplementedError


class Krum(Aggregator):
    """multi-KRUM."""

    def __init__(self, num_byzantine: int, num_selected: int = 1):
        super().__init__()
        assert num_byzantine >= 0 and num_selected >= 1
        self.num_byzantine = num_byzantine
        self.num_selected = num_selected
    
    @classmethod
    def with_learning_settings(cls, batch_size: int, alpha: float) -> "Krum":
        """Configure multi-KRUM in byzantine machine learning.

        Parameters:
            batch_size (int): the clean data batch size.
                **Warning** : this is not the true batch size since it does not
                include poisons, the latter are artificially inserted into the batch.
            alpha (float): the maximum proportion of poisons to defend against.

        Returns:
            aggregator (Krum): a robust aggregator.
        """
        alpha = 0.1
        f = int(alpha / (1 - alpha) * batch_size)
        n = batch_size + f
        m = n - (2 * f + 3)
        return Krum(f, m)
    
    def weights(self, matrix: Tensor) -> Tensor:
        assert matrix.shape[0] >= self.num_byzantine + 3
        assert matrix.shape[0] >= self.num_selected

        distances = torch.cdist(matrix, matrix, compute_mode="donot_use_mm_for_euclid_dist")
        n_closest = matrix.shape[0] - self.num_byzantine - 2
        smallest_distances, _ = torch.topk(distances, k=n_closest + 1, largest=False)
        smallest_distances_excluding_self = smallest_distances[:, 1:]
        scores = smallest_distances_excluding_self.sum(dim=1)

        _, selected_indices = torch.topk(scores, k=self.num_selected, largest=False)
        one_hot_selected_indices = F.one_hot(selected_indices, num_classes=matrix.shape[0])
        weights = one_hot_selected_indices.sum(dim=0).to(dtype=matrix.dtype) / self.num_selected

        return weights
    
    def forward(self, matrix: Tensor) -> Tensor:
        return self.weights(matrix) @ matrix
    
    def __repr__(self):
        return f"Krum(num_byzantine={self.num_byzantine}, num_selected={self.num_selected})"


# TODO: improve memory consumption (delete original tensors?)
def combine_jacobians(jacs: list[Tensor]) -> Tensor:
    """Combine a list of unflattened jacobians into a single jacobian matrix.
    
    Parameters:
        jacs (list of Tensor): a list of parameter jacobians. All of these jacobians
            must have their first dimension length equal to the batch size `B`.
    
    Returns:
        matrix (Tensor): a combined 2D jacobian matrix of dimension `B x D` where
            `B` is the batch size and `D` is the length of a single flattened gradient.
    """
    return torch.cat([jac.flatten(start_dim=1) for jac in jacs], dim=1)

def uncombine_gradients_like(combined_grad: Tensor, like_jacs: list[Tensor]) -> list[Tensor]:
    """Split and reshape a combined gradient into a list of parameter gradients
    with shapes given by a list of parameter jacobians.

    Parameters:
        combined_grad (Tensor): the combined gradient, possibly obtained by
            combining and aggregating `like_jacs`.
        like_jacs (list of Tensor): reshape like this list of jacobians
            (ignoring the batch size).
    
    Returns:
        gradients (list of Tensor): a list of parameter gradients.    
    """
    param_shapes = [jac.shape[1:] for jac in like_jacs]
    # Divide by the batch size
    param_lengths = [jac.numel() // jac.shape[0] for jac in like_jacs]
    return [
        grad.reshape(shape)
        for grad, shape in zip(combined_grad.split(param_lengths), param_shapes)
    ]    


def aggregate(jacs: list[Tensor], aggregator: Aggregator) -> list[Tensor]:
    """Aggregate per-sample gradients.

    Parameters:
        jacs (list of Tensor): a list of parameter jacobians, i.e a list of tensors
            with their first dimension equal to the batch size and the other dimensions
            with the same shape as the parameters.
        aggregator (Aggregator): the gradient aggregation method.

    Returns:
        grads (list of Tensor): a list of parameter gradients.

    Example:
    ```python
    grads = grad_batched(losses, model)
    grads = aggregate(grads)
    ```
    """
    assert all([jac.shape[0] == jacs[0].shape[0] for jac in jacs])

    combined_jacs = combine_jacobians(jacs)

    combined_grad = aggregator(combined_jacs)
    del combined_jacs
    
    return uncombine_gradients_like(combined_grad, jacs)

# https://pytorch.org/tutorials/intermediate/per_sample_grads.html
def per_sample_grads(
        model: nn.Module,
        X: Tensor,
        y: Tensor,
        criterion: _Loss,
        store_in_params: bool = False,
        append: bool = True,
    ) -> dict[str, Tensor]:
    """Compute per-sample gradients (jacobians).

    Parameters:
        model (Module): a neural network that takes a batch in argument.
        X (Tensor): the batch inputs.
        y (Tensor): the batch targets.
        criterion (loss): the loss function.
        store_in_params (bool, defaults to False): whether to store
            the per-sample gradients in the parameters `.jac` field.
            Ensure that `set_jacs_to_none` is called in order to avoid memory leaks.
        append (bool, defaults to False): whether to append additional rows
            to existing jacobian matrices stored in the `.jac` fields.

    Returns:
        jacs (dict): a dict of the gradients indexed by the parameter names.
    
    **Important note**: the model must have `track_running_stats=False`
    for each batch normalization layer to be compatible with `torch.func` AD.
    This can be done automatically by calling `utils.disable_bn_modules` on your model.
    However, disabling BN stats will incur a large drop in accuracy and will make
    the model accuracy dependent on a single batch size. Alternatively, consider
    replacing `BatchNorm` with `GroupNorm` using `utils.convert_bn_modules_to_gn`.
    This change in architecture incurs a medium drop in speed but makes training
    more robust to non i.i.d data and small batch sizes.

    Example:
    ```python
    jacs = per_sample_grads(model, X, y, criterion)
    # Manual aggregation
    for (name, param) in model.named_parameters():
        param.grad = jacs[name].mean(dim=0)
    ```
    """

    # TODO: don't detach for performance? relevant with large models
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    # A functional that computes the loss on a single sample, given model parameters.
    def compute_loss(params: dict, buffers: dict, sample: Tensor, target: Tensor) -> Tensor:
        # Unsqueeze since the model and the loss expect batches
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = ft.functional_call(model, (params, buffers), (batch,))

        loss = criterion(predictions, targets)
        if tuple(loss.shape) == (1,):
            # Just in case loss.reduction == 'none'
            loss = loss.squeeze()
        else:
            assert len(loss.shape) == 0
        
        return loss

    # A function that computes the loss gradients
    ft_compute_grad = ft.grad(compute_loss)
    # A function that computes the per-sample gradients on a batch
    ft_compute_sample_grad = ft.vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

    jacs: dict = ft_compute_sample_grad(params, buffers, X, y)

    if store_in_params:
        for (param, jac) in zip(model.parameters(), jacs.values()):
            # TODO: detach_() to save memory??
            if append and (
                    hasattr(param, 'jac') and
                    isinstance(param.jac, Tensor) and
                    param.jac.ndim == param.ndim + 1
                ):
                # Append to the jacobian matrix
                param.jac = torch.cat([param.jac, jac])
            else:
                # Store the jacobian matrix in the grad field
                param.jac = jac

    return jacs

def backpropagate_grads(
        model: nn.Module,
        X: Tensor, y: Tensor,
        criterion: _Loss, aggregator: Aggregator,
    ):
    """Backpropagate and aggregate model gradients on a mini-batch.

    This method is equivalent to calling `.backward()` on the computed loss,
    but allows any gradient aggregation method.

    Parameters:
        model (Module): a neural network that takes a batch in argument.
        X (Tensor): the batch inputs.
        y (Tensor): the batch targets.
        criterion: the loss function.
        aggregator (Aggregator): the gradient aggregation method.
    
    **Important note on batch normalization**: see `per_sample_grads`
    for the requirement on `model`.
    """
    per_sample_grads(model, X, y, criterion, store_in_params=True)
    aggregate_and_store_grads(model, aggregator)

def aggregate_and_store_grads(model: nn.Module, aggregator: Aggregator):
    """Aggregate the per-sample model gradients and store them. 

    Assumes that the jacobians have already been stored in the model with
    `per_sample_grads(..., store_in_params=True)`. This also sets the parameters
    `.jac` fields to `None`.

    Args:
        model (Module): a neural network
        aggregator (Aggregator): the gradient aggregation method.
    """
    agg_grads = aggregate([param.jac for param in model.parameters()], aggregator)

    # The parameter order is preserved by dict
    for (param, grad) in zip(model.parameters(), agg_grads):
        param.jac = None
        param.grad = grad

def set_jacs_to_none(x: Tensor | nn.Module):
    """Delete the per-sample gradients of a tensor or a model."""
    if isinstance(x, nn.Module):
        for param in x.parameters():
            set_jacs_to_none(param)
    elif isinstance(x, Tensor):
        if hasattr(x, 'jac'):
            x.jac = None
    else:
        raise TypeError(x.__class__.__name__)
