from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from random import randint
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss, CrossEntropyLoss
from torch.optim import Optimizer, SGD, Adam, AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.functional import total_variation
import torchinfo

from .nn import _detect_device
from .utils import trange
from .datasets import EagerDataset



class GradientAttack(Enum):
    """
    A type of gradient attack.
    """
    # Inverting gradient reconstruction attack (Geiping et al., 2020)
    # https://arxiv.org/abs/2003.14053v2)
    RECONSTRUCTION = -1

    # Gradient Ascent (Blanchard et al., 2017)
    ASCENT = 0

    # Orthogonal Gradient
    ORTHOGONAL = 1

    # Little is Enough (Baruch et al., 2019; Shejwalkar & Houmansadr, 2021)
    # https://arxiv.org/abs/1902.06156
    LITTLE_IS_ENOUGH = 2



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
    """A class for estimating model gradient statistics (mean and standard deviation)."""
    def average_clean_gradient(self, model: nn.Module, criterion: _Loss) -> Tensor:
        """
        Estimate the average gradient on a clean-distributed dataset.

        # Requirements
        This function must not modify the model gradients.
        """
        raise NotImplementedError
    
    def std_clean_gradient(self, model: nn.Module, criterion: _Loss) -> Tensor:
        """
        Estimate the gradient per-coordinate standard deviation on a clean-distributed dataset.
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

    def std_clean_gradient(self, model: nn.Module, criterion: _Loss) -> Tensor:
        # FIXME: gradients have already been aggregated...
        raise NotImplementedError
    

class ShadowGradientEstimator(GradientEstimator):
    """
    Estimate the average clean gradient with an auxiliary dataset
    that is similarly distributed to the training dataset.
    """
    def __init__(self, aux_loader: DataLoader):
        self.aux_loader = cycle(aux_loader)
    
    def average_clean_gradient(self, model: nn.Module, criterion: _Loss) -> Tensor:
        model = deepcopy(model)
        # TODO: estimate on next mini-batch of aux_loader
        # WARNING: do not modify the original model, but the copied model!
        #
        # minor suggestion (optimization): identify samples that are consistently
        # close to the average and boost them.
        # also cache near-constant gradients if model is converging
        raise NotImplementedError
    
    def std_clean_gradient(self, model: nn.Module, criterion: _Loss):
        model = deepcopy(model)
        # FIXME: gradients have already been aggregated...
        raise NotImplementedError


class SampleInit:
    """Sample initialization method for inverting gradient attacks."""
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __call__(self) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

class SampleInitRandomNoise(SampleInit):
    """Generate an image with random noise and a random label."""
    def __call__(self) -> tuple[Tensor, Tensor]:
        return self.dataset.random_sample_noise()

# FIXME: this would leak training data to the attacker.
# TODO: sample from auxiliary dataset instead
class SampleInitFromDataset(SampleInit):
    """Choose a random image from the dataset."""
    def __call__(self) -> tuple[Tensor, Tensor]:
        return self.dataset[randint(len(self.dataset))]

class SampleInitConstant(SampleInit):
    """Return a fixed starting image."""
    def __init__(self, X: Tensor, y: Tensor):
        self.X = X
        self.y = y
    
    def __call__(self) -> tuple[Tensor, Tensor]:
        return self.X, self.y


class Schedule(ABC):
    """An update schedule for the attack steps.
    
    In the gradient inverting attack, the poisoned label is updated according to this schedule.
    """
    @abstractmethod
    def __call__(self, step: int) -> bool:
        """Returns whether an update should be performed at this step."""

class PowerofTwoSchedule(Schedule):
    """An exponential schedule that fires every step that is a power of two.
    
    The update frequency decreases exponentially to ensure proper convergence.
    This is necessary since label flipping is not a continuous operation.
    """
    def __call__(self, step: int) -> bool:
        return step & (step - 1) == 0


class NeverUpdate(Schedule):
    """Never update the label."""
    def __call__(self, _step: int):
        return False


class GradientInverter:
    """Inverting gradient attack."""
    def __init__(
            self,
            method: GradientAttack,
            estimator: GradientEstimator,
            steps: int,
            sample_init: SampleInit,
            tv_coef = 0.0, # TODO: more ergonomic interface
            lr = 0.2, # TODO: find appropriate lr scheduling step
            label_update_schedule: Schedule = PowerofTwoSchedule(),
        ):
        self.method = method
        self.estimator = estimator
        self.steps = steps
        self.tv_coef = tv_coef
        self.lr = lr
        self.sample_init = sample_init
        self.label_update_schedule = deepcopy(label_update_schedule)
    
    def attack_objective(
            self,
            model_grad: Tensor,
            avg_clean_grad: Tensor,
            x_base: Tensor = None,
        ) -> Tensor:
        """Returns the adversary's objective to minimize."""
        # FIXME: not reliable
        training_data: EagerDataset = self.sample_init.dataset
        max_data_variation = training_data.max_data_variation()

        match self.method:
            case GradientAttack.RECONSTRUCTION:
                #TODO: compare cos_sim with distance squared
                #TODO: (signed gradient updates) & learning rate decay (Geiping et al.)
                loss_atk = 1.0 - torch.cosine_similarity(model_grad, avg_clean_grad, dim=0)

            case GradientAttack.ASCENT:
                cos_sim = torch.cosine_similarity(model_grad, avg_clean_grad, dim=0)
                # dot product increases the gradient size but makes unalignment easier
                #loss_atk = g_p.dot(avg_clean_gradient)
                loss_atk = cos_sim
            
            case GradientAttack.ORTHOGONAL:
                cos_sim = torch.cosine_similarity(model_grad, avg_clean_grad, dim=0)
                loss_atk = cos_sim ** 2

            case GradientAttack.LITTLE_IS_ENOUGH:
                # See Algorithm 3 in https://arxiv.org/pdf/1902.06156
                # Estimate per-coordinate gradient std dev. with GradientEstimator
                raise NotImplementedError

        if self.tv_coef:
            assert x_base is not None
            tv = total_variation(x_base.unsqueeze(0))
            normalization = x_base.numel() * 4 * max_data_variation ** 2
            loss_atk += self.tv_coef * tv / normalization
        
        return loss_atk
    
    def attack(self, model: nn.Module, criterion: _Loss) -> tuple[Tensor, Tensor]:
        """
        Create a poisoned data point with an inverting gradient attack.
        
        This function assumes that mini-batch loss gradients have been computed
        via backpropagation. It does not alter the model.

        WARNING: do not deepcopy the model before calling this function
        as the gradients will not be copied along.
        """
        # FIXME: not reliable
        training_data: EagerDataset = self.sample_init.dataset
        num_classes = len(training_data.classes)
        device = _detect_device(model)

        avg_clean_gradient = self.estimator.average_clean_gradient(model, criterion)
        avg_clean_gradient.requires_grad_(False)
        
        # This detaches the model and its gradients
        model = deepcopy(model)
        model.eval()
        model.requires_grad_()
        model.zero_grad()

        x_base, y_base = self.sample_init()
        
        x_base = x_base.to(device)
        y_base = F.one_hot(y_base, num_classes).float().to(device)
        # We optimize on the image...
        opt = Adam([x_base], lr=self.lr)
        # ...and occasionally on the label.

        for step in range(self.steps):
            x_base.requires_grad_(True)
            y_base.requires_grad_(True)

            # --I--
            y_pred = model(x_base.unsqueeze(0)).squeeze()
            loss = criterion(y_pred, y_base)
            loss.backward(create_graph=True) # Allows 2nd-order differentiation
            
            loss_atk = self.attack_objective(
                model_gradients(model),
                avg_clean_gradient,
                x_base,
            )
            #print(f"Inverting gradient step {step}: loss = {loss.item()}, loss_atk = {loss_atk.item()}")
            # Clear `loss` gradients on `x_base`
            opt.zero_grad()

            # TODO: improve efficiency, e.g with torch.autograd
            # Optimize `x_base`
            loss_atk.backward()
            opt.step()
            opt.zero_grad()
            model.zero_grad()

            # Avoids autograd graph errors when modifying tensor in-place
            x_base.requires_grad_(False)
            # Projected optimization algorithm
            training_data.clip_to_data_range(x_base, inplace=True)
            
            # --II---
            if self.label_update_schedule(step):
                # Instead of optimal label flipping (as in https://arxiv.org/abs/2503.00140),
                # we guess the best class based on `loss_atk` gradients w.r.t y
                # Efficiency is key when the number of classes grows
                y_candidate = F.one_hot(y_base.grad.argmin(), num_classes).float()
                y_pred = model(x_base.unsqueeze(0)).squeeze()
                loss = criterion(y_pred, y_candidate)
                loss.backward()
                loss_atk_2 = self.attack_objective(
                    model_gradients(model),
                    avg_clean_gradient,
                    x_base,
                )
                # Change the class if it improves the adversary's objective
                if loss_atk_2 < loss_atk:
                    y_base = y_candidate
        
            y_base.requires_grad_(False)
            y_base.grad = None
            model.zero_grad()

        y_base = y_base.argmax()
        return x_base, y_base


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