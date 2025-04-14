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
    ) -> dict[str, Tensor]:
    """Compute per-sample gradients (jacobians).

    Parameters:
        model (Module): a neural network that takes a batch in argument.
        X (Tensor): the batch inputs.
        y (Tensor): the batch targets.
        criterion (loss): the loss function.
        store_in_params (bool, defaults to False): whether to store
            the per-sample gradients in the parameters `.grad` field.

    Returns:
        grads (dict): a dict of the gradients indexed by the parameter names.
    
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
    grads = per_sample_grads(model, X, y, criterion)
    # Manual aggregation
    for (name, param) in model.named_parameters():
        param.grad = grads[name].mean(dim=0)
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

    grads: dict = ft_compute_sample_grad(params, buffers, X, y)

    if store_in_params:
        for (param, grad) in zip(model.parameters(), grads.values()):
            # Store the jacobian matrix in the grad field
            # TODO: detach_() to save memory??
            param.grad = grad

    return grads


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
    grads = per_sample_grads(model, X, y, criterion)
    agg_grads = aggregate(list(grads.values()), aggregator)

    # The parameter order is preserved by dict
    for (param, grad) in zip(model.parameters(), agg_grads):
        param.grad = grad


@deprecated("This is an internal test only intended for early iterations of the code.")
def _print_autojac_error(
        model: nn.Module,
        X: Tensor, y: Tensor,
        criterion: _Loss,
    ):
    """Print the errors of `per_sample_grads` compared to naive repeated AD on each sample.

    This prints the accumulated errors for each parameter group.
    """

    grads = per_sample_grads(model, X, y, criterion)
    manual_grads = compute_sample_grads_naive(model, X, y, criterion)
    for jac, ft_jac in zip(manual_grads, grads.values()):
        # NOTE: there is a ~1e-5 error in L^inf norm -> ~10 error in L^1
        # This approximation error is hard to fix but does not seem to impact performance.
        if not torch.allclose(jac, ft_jac, rtol=1e-1):
            print('Error:')
            print('grad shape', jac.shape)
            print('cos_sim', F.cosine_similarity(jac.flatten(), ft_jac.flatten(), dim=0))
            print('L^1', (jac - ft_jac).norm(1))
            print('L^inf', (jac - ft_jac).abs().max())
            print(
                'max_rtol',
                ((jac - ft_jac) / (jac + 1e-16)).abs().max()
            )


@deprecated("Naive gradient function. Use `per_sample_grads` instead.")
def compute_sample_grads_naive(model, data, targets, criterion):
    """ manually process each sample with per sample gradient """
    def _compute_grad_naive(model, sample, target, criterion):
        sample = sample.unsqueeze(0)  # prepend batch dimension for processing
        target = target.unsqueeze(0)

        prediction = model(sample)
        loss = criterion(prediction, target)

        return torch.autograd.grad(loss, list(model.parameters()))

    sample_grads = [
        _compute_grad_naive(model, data[i], targets[i], criterion)
        for i in range(len(data))
    ]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads

@deprecated("Naive backward. Consider using `backpropagate_gradients` which is faster.")
def naive_backpropagate_grads(model, data, targets, criterion, aggregator: Aggregator):
    grads = compute_sample_grads_naive(model, data, targets, criterion)
    agg_grads = aggregate(grads, aggregator)
    for (param, grad) in zip(model.parameters(), agg_grads):
            param.grad = grad


@deprecated("Consider using `per_sample_grads` which is faster, though less flexible.")
def grad_batched(losses: Tensor, model: nn.Module) -> list[Tensor]:
    """Returns the losses gradients w.r.t the model parameters in batches.
        
    Parameters:
        losses (Tensor): the losses computed on a batch.
            Set `reduce='none'` to the loss parameters to compute this quantity.
        model (Module): the neural network.

    Returns:
        grads (list of Tensor): a list of parameter gradients with their first dimension
            equal to the batch size.

    Example:
    ```python
    criterion = CrossEntropyLoss(reduction='none') # Disable mean reduction
    losses = criterion(model(X), y)
    grads = grad_batched(losses, model)
    ```
    """
    # Jacobian backpropagation.

    # UserWarning: There is a performance drop because we have not yet implemented
    # the batching rule for aten::nll_loss_backward.
    # You are using the legacy vmap prototype (torch._vmap_internals.vmap).
    # If you are using torch.autograd.functional.{jacobian, hessian} or torch._vmap_internals.vmap:
    # please switch to using torch.func.{jacrev, jacfwd, hessian} and/or torch.vmap instead
    # for better operator coverage and performance improvements.
    torch._C._debug_only_display_vmap_fallback_warnings(True)

    vs = torch.eye(losses.shape[0], device=losses.device)
    return torch.autograd.grad(losses, model.parameters(), vs, is_grads_batched=True)

    #def _grad(loss: Tensor):
    #    return torch.autograd.grad(loss, model.parameters(), retain_graph=True)
    #torch.func.grad
    #return torch.vmap(_grad)(losses)


@deprecated("Consider using `backpropagate_gradients` which is faster, though less flexible.")
def backward(losses: Tensor, model: nn.Module, aggregator: Aggregator):
    """Perform backpropagation.

    Parameters:
        losses (Tensor): the losses computed on a batch.
            Set `reduce='none'` to the loss parameters to compute this quantity.
        model (Module): the neural network.
        aggregator (Aggregator): the gradient aggregation method.
    Example:
        ```python
        backward(losses, model.parameters(), aggregator='mean')
        ```
    """
    grads = grad_batched(losses, model)
    agg_grads = aggregate(grads, aggregator)
    for (param, grad) in zip(model.parameters(), agg_grads):
            param.grad = grad
