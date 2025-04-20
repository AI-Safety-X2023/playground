from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from random import randint
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import total_variation

import federated as fed

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


class GradientEstimator(ABC):
    """A class for estimating model gradient statistics (mean and standard deviation)."""
    
    @abstractmethod
    def average_clean_gradient(self, model: nn.Module, criterion: _Loss) -> Tensor:
        """Estimate the average gradient on a clean-distributed dataset.

        ## Requirements
        This function must not modify the model gradients.
        """

    @abstractmethod    
    def std_clean_gradient(self, model: nn.Module, criterion: _Loss) -> Tensor:
        """
        Estimate the gradient per-coordinate standard deviation on a clean-distributed dataset.
        """


class OmniscientGradientEstimator(GradientEstimator):
    """Estimates the average gradient assuming the per-sample gradients have
    already been computed on a mini-batch with loss backpropagation.

    ## Examples
    
    Computing the mean:
    ```python
    estimator = OmniscientGradientEstimator()
    X, y = next(self.aux_loader_iter)
    loss = criterion(model(X), y)
    loss.backward()
    avg_clean_grad = estimator.average_clean_gradient(model, criterion)
    ```
    Computing the standard deviation:
    ```python
    estimator = OmniscientGradientEstimator()
    federated.per_sample_grads(model, X, y, criterion, store_in_params=True)
    std_clean_grad = estimator.std_clean_gradient(model, criterion)
    ```
    """
    
    # NOTE: Assumes that `criterion.reduction == 'mean'`
    def average_clean_gradient(self, model: nn.Module, criterion: _Loss) -> Tensor:
        if all(param.grad.shape == param.shape for param in model.parameters()):
            # Gradients are already accumulated. We assume mean reduction.
            return combined_model_gradients(model)

        # Otherwise, aggregate jacobians
        aggregator = fed.Mean()
        jacs = [param.jac for param in model.parameters()]
        assert all(jac.shape[0] == jacs[0].shape[0] for jac in jacs)
        matrix = fed.combine_jacobians(jacs)
        return aggregator(matrix)

    def std_clean_gradient(self, model: nn.Module, criterion: _Loss) -> Tensor:
        """Assumes jacobians have been stored with `federated.per_sample_grads`."""
        aggregator = fed.Stddev()
        jacs = [param.jac for param in model.parameters()]
        matrix = fed.combine_jacobians(jacs)
        return aggregator(matrix)
    

class ShadowGradientEstimator(GradientEstimator):
    """
    Estimate the average clean gradient with an auxiliary dataset
    that is similarly distributed to the training dataset.
    """
    def __init__(self, aux_loader: DataLoader):
        self.aux_loader_iter = iter(cycle(aux_loader))
    
    # NOTE: Assumes that `criterion.reduction == 'mean'`
    def average_clean_gradient(self, model: nn.Module, criterion: _Loss) -> Tensor:
        # Copy the model since we modify its gradients
        model = deepcopy(model)
        X, y = next(self.aux_loader_iter)
        loss = criterion(model(X), y)
        loss.backward()
        return combined_model_gradients(model)

        # minor suggestion (optimization): identify samples that are consistently
        # close to the average and boost them.
        # also cache near-constant gradients if model is converging
    
    def std_clean_gradient(self, model: nn.Module, criterion: _Loss):
        model = deepcopy(model)
        X, y = next(self.aux_loader_iter)
        fed.per_sample_grads(model, X, y, criterion, store_in_params=True)
        std = OmniscientGradientEstimator().std_clean_gradient(model, criterion)
        fed.set_jacs_to_none(model)
        return std


class SampleInit(ABC):
    """Sample initialization method for inverting gradient attacks."""
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @abstractmethod
    def __call__(self) -> tuple[Tensor, Tensor]:
        """Returns the initial input and label."""


class SampleInitRandomNoise(SampleInit):
    """Generate an image with random noise and a random label."""
    def __call__(self) -> tuple[Tensor, Tensor]:
        return self.dataset.random_sample_noise()


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


class LearningSettings:
    """The targeted learning algorithm and threat model settings.

    We consider the learning setting given by Bouaziz et al. (2024):
    Inverting Gradients Makes Powerful Data Poisoning [1].

    For now, we constrain ourselves to *FedSGD* with data poisoning:
    - At every step, the workers send one data point each.
    - The central machine computes and aggregates the gradients itself.
    - Some of the workers are "byzantine", controlled by a single attacker who can send
        arbitrary valid data to the central machine.

    *FedSGD* is an equivalent version of centralized learning. In this context, the data
    suppliers may be malicious, but the reported gradients are always honest, contrary
    to vanilla gradient attacks.
    For now, we do not consider other federated learning settings such as *FedAvg*
    and *LocalSGD*.

    The attacker has knowledge of all of these learning settings and uses them
    to perform the attack. In particular, LIE requires the full knowledge as described
    by this class.

    [1]: https://arxiv.org/abs/2410.21453
    """
    # TODO: this class describes the attacker's power and its fixed knowledge.
    # We can also add variable knowledge (model weights, batch or aux. dataset)
    # This would be a partially stateless knowledge module like `GradientEstimator`.
    def __init__(
            self,
            criterion: _Loss,
            aggregator: fed.Aggregator = None,
            num_clean: int = None,
            num_byzantine: int = None,
        ):
        """Initialize the target settings.

        Parameters:
            criterion (_Loss or function): the loss function used by the defender.
            aggregator (Aggregator): the aggregator used by the defender.
                Defaults to `None`. Required for the LIE attack.
            num_clean (int): the clean batch size, i.e the number of clean machines.
                Defaults to `None`. Required for the LIE attack.
            num_byzantine (int): the number of poisoned machines.
                Defaults to `None`. Required for the LIE attack.
        
        :note: `num_byzantine` is the true number of byzantine suppliers, which is usually
        different from `Krum.num_byzantine` in general, since the latter is the maximum
        number of byzantine suppliers to defend against.
        """
        self.criterion = criterion
        self.aggregator = aggregator
        self.num_clean = num_clean
        self.num_byzantine = num_byzantine
    
    @property
    def poison_factor(self) -> float:
        """The proportion of poisoned machines owned by the attacker."""
        return self.num_byzantine / (self.num_clean + self.num_byzantine)


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
    
    def gradient_objective(
            self,
            model_grad: Tensor,
            target_grad: Tensor,
            x_p: Tensor = None,
        ) -> Tensor:
        """Returns the adversary's objective to minimize."""
        # FIXME: not reliable
        training_data: EagerDataset = self.sample_init.dataset
        max_data_variation = training_data.max_data_variation()

        match self.method:
            case GradientAttack.RECONSTRUCTION:
                #TODO: compare cos_sim with distance squared
                #TODO: (signed gradient updates) & learning rate decay (Geiping et al.)
                loss_atk = 1.0 - torch.cosine_similarity(model_grad, target_grad, dim=0)

            case GradientAttack.ASCENT:
                cos_sim = torch.cosine_similarity(model_grad, target_grad, dim=0)
                # dot product increases the gradient size but makes unalignment easier
                #loss_atk = g_p.dot(avg_clean_gradient)
                loss_atk = cos_sim
            
            case GradientAttack.ORTHOGONAL:
                cos_sim = torch.cosine_similarity(model_grad, target_grad, dim=0)
                loss_atk = cos_sim ** 2

            case GradientAttack.LITTLE_IS_ENOUGH:
                loss_atk = F.mse_loss(model_grad, target_grad)

        if self.tv_coef:
            assert x_p is not None
            tv = total_variation(x_p.unsqueeze(0))
            normalization = x_p.numel() * 4 * max_data_variation ** 2
            loss_atk += self.tv_coef * tv / normalization
        
        return loss_atk
    
    def poison_objective(
            self,
            x_p: Tensor, y_p: Tensor, target_grad: Tensor,
            model: nn.Module, criterion: _Loss,
            differentiable: bool = False,
        ):
        y_pred = model(x_p.unsqueeze(0)).squeeze()
        loss = criterion(y_pred, y_p)
        loss.backward(create_graph=differentiable)
        return self.gradient_objective(
            combined_model_gradients(model),
            target_grad,
            x_p,
        )
    
    def _lie_find_std_factor(
            self,
            settings: LearningSettings,
            jac_matrix: Tensor, avg_grad: Tensor, std_grad: Tensor,
            z_init = 1.0, threshold = 0.1, # TODO: adjust and justify these values
        ) -> Tensor:
        """Find the best factor to deviate the (estimated) average gradient
        for Little Is Enough attack. This is the gradient attack specified in Algorithm 1
        of Shejwalkar and Houmansadr, *Manipulating the Byzantine* (2021) [1]

        Returns:
            z (Tensor): A single positive floating-point factor.
        
        [1] <https://www.semanticscholar.org/paper/be10a3afb028e971f38fa80347e4bd826724b86a>
        """
        aggregator = settings.aggregator
        clean_batch_size = settings.num_clean
        num_harmful = settings.num_byzantine

        # Expand Jacobian matrix with the harmful gradients
        jac_matrix = torch.cat((jac_matrix, torch.zeros(num_harmful, jac_matrix.shape[1])))

        # TODO: Algorithm 1 in
        # https://www.semanticscholar.org/paper/be10a3afb028e971f38fa80347e4bd826724b86a
        z = z_init
        z_succ = float('inf')
        step = z_init / 2
        while abs(z_succ - z) > threshold:

            # Update the poisoned gradients in the Jacobian matrix
            poisoned_grads = (avg_grad - z * std_grad).unsqueeze(0).expand(num_harmful, -1) #pour etre en [1,D] puis [num_harmful,D]
            jac_matrix[clean_batch_size:] = poisoned_grads

            # Compute the number of gradients selected by the aggregator
            if isinstance(aggregator, fed.Krum):
                selection = aggregator.weights(jac_matrix)
                num_selected = selection[clean_batch_size:].sum()
                # TODO
            elif isinstance(aggregator, fed.Mean):
                raise NotImplementedError(
                    "We do not implement LIE against non-robust gradient aggregation "
                    "since the LIE optimization problem is unbounded for `Mean`."
                )
            else:
                raise RuntimeError(f"Unknown aggregator: {aggregator.__class__}")
            
            if num_selected >= 1 :
                z_succ = z
                z = z + step / 2
            else:
                z = z - step / 2
            step = step / 2
        return z_succ

    def lie_attack(self, model: nn.Module, settings: LearningSettings) -> Tensor:
        """Find the target gradient for the Little Is Enough attack.

        Returns:
            target_grad (Tensor): the target gradient.
        """
        clean_batch_size = settings.num_clean

        # Get the per-sample clean gradients
        if isinstance(self.estimator, OmniscientGradientEstimator):
            # The true per-sample gradients are already stored in model parameters .jac field
            assert all(
                clean_batch_size == param.grad.shape[0]
                for param in model.parameters()
            )
            jacs = [param.jac for param in model.parameters()]
        elif isinstance(self.estimator, ShadowGradientEstimator):
            X, y = next(self.estimator.aux_loader_iter)
            # Compute some shadow per-sample gradients
            jacs = list(fed.per_sample_grads(model, X, y, settings.criterion).values())
            assert clean_batch_size == len(X)

        jac_matrix = fed.combine_jacobians(jacs)

        avg_grad = fed.Mean()(jac_matrix).requires_grad_(False)
        std_grad = fed.Stddev()(jac_matrix).requires_grad_(False)

        z = self._lie_find_std_factor(settings, jac_matrix, avg_grad, std_grad)
        
        # Calculate the final gradient of the attack (LIE)
        return avg_grad + z * std_grad
    
    def attack(
            self,
            model: nn.Module,
            criterion: _Loss = None,
            settings: LearningSettings = None,
        ) -> tuple[Tensor, Tensor]:
        """Create a poisoned data point with an inverting gradient attack.
        
        This function assumes that batch loss jacobians have been computed
        via backpropagation. It does not alter the model.

        **WARNING**: do not deepcopy the model before calling this function
        as the gradients will not be copied along.

        Parameters:
            model (Module): the targeted neural network weights at the current step.
            criterion (_Loss, optional): the defender's loss function. Mutually exclusive
                with :param:`settings`
            settings (LearningSettings, optional): the learning settings and threat model.
                Required for *Little Is Enough*.
        
        Returns:
            x_y (tuple): the crafted poison to be used by all malicious workers.

        Examples:
        For all attacks except *Little Is Enough*:
        ```python
        inverter = GradientInverter(
            GradientAttack.ASCENT,
            OmniscientGradientEstimator(),
            steps=5,
            sample_init=SampleInitRandomNoise(aux_data),
        )
        criterion(model(X), y).backward()
        X_p, y_p = inverter.attack(model, criterion)
        ```
        For *Little Is Enough*:
        ```python
        inverter = GradientInverter(
            GradientAttack.LITTLE_IS_ENOUGH,
            OmniscientGradientEstimator(),
            steps=5,
            sample_init=SampleInitRandomNoise(aux_data),
        )
        per_sample_grads(model, X, y, criterion, store_in_params=True)
        X_p, y_p = inverter.attack(model, settings)
        ```
        """
        # FIXME: not reliable
        training_data: EagerDataset = self.sample_init.dataset
        num_classes = len(training_data.classes)
        device = _detect_device(model)

        if self.method == GradientAttack.LITTLE_IS_ENOUGH:
            # TODO: report number of selected gradients
            target_grad = self.lie_attack(model, settings)
        else:
            # TODO: log gradient estimation quality
            target_grad = self.estimator.average_clean_gradient(model, criterion)
        target_grad.requires_grad_(False)
        
        # This detaches the model and its gradients
        model = deepcopy(model)
        model.eval()
        model.requires_grad_()
        model.zero_grad()

        x_p, y_p = self.sample_init()
        
        x_p = x_p.to(device)
        y_p = F.one_hot(y_p, num_classes).float().to(device)
        # We optimize on the image...
        opt = Adam([x_p], lr=self.lr)
        # ...and occasionally on the label.

        for step in range(self.steps):
            x_p.requires_grad_(True)
            y_p.requires_grad_(True)

            # --I--
            loss_atk = self.poison_objective(
                x_p, y_p, target_grad,
                model, criterion, differentiable=True,
            )
            # TODO: log stats
            #print(f"Inverting gradient step {step}: loss = {loss.item()}, loss_atk = {loss_atk.item()}")
            # Clear `loss` gradients on `x_p`
            opt.zero_grad()

            # Optimize `x_p`
            loss_atk.backward(inputs=x_p)
            opt.step()
            opt.zero_grad()
            model.zero_grad()

            # Avoids autograd graph errors when modifying tensor in-place
            x_p.requires_grad_(False)
            # Projected optimization algorithm
            training_data.clip_to_data_range(x_p, inplace=True)
            
            # --II---
            if self.label_update_schedule(step):
                # Instead of optimal label flipping (as in https://arxiv.org/abs/2503.00140),
                # we guess the best class based on `loss_atk` gradients w.r.t y
                # Efficiency is key when the number of classes grows
                y_candidate = F.one_hot(y_p.grad.argmin(), num_classes).float()
                loss_atk_2 = self.poison_objective(
                    x_p, y_candidate, target_grad,
                    model, criterion,
                )
                # Change the class if it improves the adversary's objective
                if loss_atk_2 < loss_atk:
                    y_p = y_candidate
        
            y_p.requires_grad_(False)
            y_p.grad = None
            model.zero_grad()

        y_p = y_p.argmax()
        return x_p, y_p


def combined_model_gradients(model: nn.Module) -> Tensor:
    """
    Returns the model gradients without detaching them, combined into a single vector.
    """
    grads = [
        param.grad.flatten()
        for param in model.parameters()
    ]
    return torch.cat(grads)
