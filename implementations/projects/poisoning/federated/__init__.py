from abc import ABC, abstractmethod
from warnings import deprecated

import torch
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.func import functional_call, vmap, grad


# Inspired by torchjd: https://torchjd.org
class Aggregator(nn.Module, ABC):
    """An abstract class for gradient aggregators.
    
    It aggregates matrices of dimension `(b, d)` into row gradients of dimension `d`.
    """
    @abstractmethod
    def forward(self, matrix: Tensor) -> Tensor:
        """Computes the aggregation from the input matrix."""


class Mean(Aggregator):
    def forward(self, matrix: Tensor) -> Tensor:
        return matrix.mean(dim=0)

 
class Krum(Aggregator):
    def forward(self, matrix: Tensor) -> Tensor:
        raise NotImplementedError    



def aggregate(grads: list[Tensor], aggregator: Aggregator) -> list[Tensor]:
    """Aggregate batched gradients.

    Args:
        grads (list of Tensor): a list of parameter gradients with their first dimension
            equal to the batch size.
        aggregator (Aggregator): the gradient aggregation method.

    Returns:
        grads (list of Tensor): a list of parameter gradients.

    Example:
    ```python
    grads = grad_batched(losses, model)
    grads = aggregate(grads)
    ```    
    """
    assert all([grad.shape[0] == grads[0].shape[0] for grad in grads])

    # Save the parameter shapes
    param_shapes = [grad.shape[1:] for grad in grads]
    param_lengths = [grad.numel() // grad.shape[0] for grad in grads]

    # Group the gradients into a batch of vectors (gradient matrix)
    model_grad_b = torch.cat([grad.flatten(start_dim=1) for grad in grads], dim=1)

    # Aggregate the gradients
    model_grad = aggregator(model_grad_b)
    del model_grad_b
    
    # Ungroup the parameter gradients
    return [
        grad.reshape(shape)
        for grad, shape in zip(model_grad.split(param_lengths), param_shapes)
    ]


# https://pytorch.org/tutorials/intermediate/per_sample_grads.html
def per_sample_grads(model: nn.Module, X: Tensor, y: Tensor, criterion: _Loss) -> dict[str, Tensor]:
    """Compute per-sample gradients.

    Args:
        model (Module): a neural network that takes a batch in argument.
        X (Tensor): the batch inputs.
        y (Tensor): the batch targets.
        criterion: the loss function without reduction.

    Returns:
        grads (dict): a dict of the gradients indexed by the parameter names.
    
    Example:
    ```python
    grads = per_sample_grads(model, X, y, criterion)
    # Manual aggregation
    for (name, param) in model.named_parameters():
        param.grad = grads[name].mean(dim=0)
    ```
    """
    # https://pytorch.org/docs/2.6/func.batch_norm.html
    # FIXME: ensure accuracy is not impacted
    from torch.func import replace_all_batch_norm_modules_
    replace_all_batch_norm_modules_(model)

    _original_reduction = criterion.reduction
    criterion.reduction = 'none'

    # TODO: don't detach for performance?
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def compute_loss(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = functional_call(model, (params, buffers), (batch,))
        loss = criterion(predictions, targets)
        return loss.squeeze()

    ft_compute_grad = grad(compute_loss)

    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

    grads = ft_compute_sample_grad(params, buffers, X, y)

    criterion.reduction = _original_reduction
    return grads


def backpropagate_grads(
        model: nn.Module,
        X: Tensor, y: Tensor,
        criterion: _Loss, aggregator: Aggregator,
    ):
    """Backpropagate and aggregate model gradients on a mini-batch.

    This method is equivalent to calling `.backward()` on the computed loss,
    but allows any gradient aggregation method.

    Args:
        model (Module): a neural network that takes a batch in argument.
        X (Tensor): the batch inputs.
        y (Tensor): the batch targets.
        criterion: the loss function without reduction.
        aggregator (Aggregator): the gradient aggregation method.
    """
    grads = per_sample_grads(model, X, y, criterion)


    #-----
    TEST = False
    if TEST:
        manual_grads = compute_sample_grads_naive(X, y)
        for per_sample_grad, ft_per_sample_grad in zip(manual_grads, grads.values()):
            # FIXME: this fails -> incorrect gradient calculation = drop in accuracy.
            # This is not only due to batchnorm patching
            assert torch.allclose(per_sample_grad, ft_per_sample_grad, rtol=1e-1)
    #-----



    agg_grads = aggregate(list(grads.values()), aggregator)

    # The parameter order is preserved by dict
    for (param, grad) in zip(model.parameters(), agg_grads):
            param.grad = grad


def _compute_grad_naive(model, sample, target, criterion):
    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    target = target.unsqueeze(0)

    prediction = model(sample)
    loss = criterion(prediction, target)

    return torch.autograd.grad(loss, list(model.parameters()))

@deprecated("Naive gradient function. Use `per_sample_grads` instead.")
def compute_sample_grads_naive(model, data, targets, criterion):
    """ manually process each sample with per sample gradient """
    from torch.func import replace_all_batch_norm_modules_
    replace_all_batch_norm_modules_(model)
    sample_grads = [_compute_grad_naive(model, data[i], targets[i], criterion) for i in range(len(data))]
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
        
    Args:
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

    Args:
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
