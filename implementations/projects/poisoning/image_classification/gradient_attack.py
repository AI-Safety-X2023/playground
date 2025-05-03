from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from random import randint
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader
from torchmetrics import CatMetric
from torchmetrics.functional import total_variation

import federated as fed

from .nn import _detect_device, MetricLogger
from .utils import mem_aware_pdist
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

    # Min-Max (Shejwalkar & Houmansadr, 2021)
    MIN_MAX = 3

    # Min-Sum (Shejwalkar & Houmansadr, 2021)
    MIN_SUM = 4

    @property
    def is_adaptative(self):
        return self in (
            GradientAttack.LITTLE_IS_ENOUGH,
            GradientAttack.MIN_MAX,
            GradientAttack.MIN_SUM,
        )

    def __str__(self):
        return {
            GradientAttack.RECONSTRUCTION: "Reconstruction",
            GradientAttack.ASCENT: "Gradient Ascent",
            GradientAttack.ORTHOGONAL: "Orthogonal Gradient",
            GradientAttack.LITTLE_IS_ENOUGH: "Little Is Enough",
            GradientAttack.MIN_MAX: "Min-Max",
            GradientAttack.MIN_SUM: "Min-Sum",
        }[self]


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
    def per_sample_gradients(self, model: nn.Module, criterion: _Loss) -> Tensor:
        """Compute per-sample gradients on a clean-distributed dataset.
        
        Returns:
            matrix (Tensor): the jacobian matrix containing per-sample row gradients.
        """
    
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
    
    def reset(self):
        """Reset the estimator's state."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


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

    def per_sample_gradients(self, model: nn.Module, criterion: _Loss) -> Tensor:
        try:
            jacs = [param.jac for param in model.parameters()]
        except AttributeError:
            raise RuntimeError(
                "You must call `per_sample_grads()` before gathering them "
                "with an omniscient estimator"
            )
        assert all(jac.shape[0] == jacs[0].shape[0] for jac in jacs), "Inconsistent jacobian shapes"
        return fed.combine_jacobians(jacs)
    
    # NOTE: Assumes that `criterion.reduction == 'mean'`
    def average_clean_gradient(self, model: nn.Module, criterion: _Loss) -> Tensor:
        if all(param.grad.shape == param.shape for param in model.parameters()):
            # Gradients are already accumulated. We assume mean reduction.
            return combined_model_gradients(model)

        # Otherwise, aggregate jacobians
        aggregator = fed.Mean()
        matrix = self.per_sample_gradients(model, criterion)
        return aggregator(matrix)

    def std_clean_gradient(self, model: nn.Module, criterion: _Loss) -> Tensor:
        """Assumes jacobians have been stored with `federated.per_sample_grads`."""
        aggregator = fed.Stddev()
        matrix = self.per_sample_gradients(model, criterion)
        return aggregator(matrix)
    

class ShadowGradientEstimator(GradientEstimator):
    """
    Estimate the average clean gradient with an auxiliary dataset
    that is similarly distributed to the training dataset.

    This estimator works by computing an exponential moving average
    of the computed gradients. This refines the estimation towards the true mean
    and takes the model updates into account.
    """
    def __init__(self, aux_loader: DataLoader, momentum: float = 0.8):
        """Initialize the estimator.

        Args:
            aux_loader (DataLoader): the auxiliary data used for gradient estimation.
            momentum (float, optional): the gradient estimation momentum. Defaults to 0.8.
        """
        self._data_len = len(aux_loader.dataset)
        self.aux_loader_iter = iter(cycle(aux_loader))

        # Gradient moment
        self._acc_grad: Tensor = None
        self.momentum = momentum

    def per_sample_gradients(self, model: nn.Module, criterion: _Loss) -> Tensor:
        device = _detect_device(model)

        # Copy the model since we modify its gradients
        model = deepcopy(model)
        X, y = next(self.aux_loader_iter)
        X, y = X.to(device), y.to(device)

        jacs = fed.per_sample_grads(model, X, y, criterion)
        jac_matrix = fed.combine_jacobians(list(jacs.values()))
        return jac_matrix
    
    # NOTE: Assumes that `criterion.reduction == 'mean'`
    def average_clean_gradient(self, model: nn.Module, criterion: _Loss) -> Tensor:
        device = _detect_device(model)

        # Copy the model since we modify its gradients
        model = deepcopy(model)
        X, y = next(self.aux_loader_iter)
        X, y = X.to(device), y.to(device)

        loss = criterion(model(X), y)
        loss.backward()
        grad = combined_model_gradients(model)

        if self._acc_grad is not None:
            m = self.momentum
            # Compute the "average" gradient with moment
            grad_moment = m * self._acc_grad + (1 - m) * grad
        else:
            grad_moment = grad

        self._acc_grad = grad_moment.detach_().clone()
        return grad_moment
    
    def std_clean_gradient(self, model: nn.Module, criterion: _Loss):
        aggregator = fed.Stddev()
        matrix = self.per_sample_gradients(model, criterion)
        return aggregator(matrix)

    def reset(self):
        self._acc_grad = None
        self._per_sample_grads_cache = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(aux_loader = DataLoader(<data_len={self._data_len}>))"
        )
    
class SampleInit(ABC):
    """Poison sample initialization method for inverting gradient attacks."""
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @abstractmethod
    def __call__(self) -> tuple[Tensor, Tensor]:
        """Returns the initial input and label."""

    def restart(self) -> tuple[Tensor, Tensor]:
        """Reinitialize any state and return a fresh sample."""
        return self()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class SampleInitRandomNoise(SampleInit):
    """Generate an image with random noise and a random label."""
    def __call__(self) -> tuple[Tensor, Tensor]:
        return self.dataset.random_sample_noise()

class SampleInitFeedback(SampleInit):
    """Improve the poison iteratively with random restarts.

    If the model updates are small, the target gradients change little so
    a suitable poison may be used a few times in a row.

    The stored poison is reset with random noise every some steps to avoid
    overfitting on similar samples (so the gradients would be too small).
    
    ## Example

    ```python
    sample_init = SampleInitFeedback(dataset, restart_period=5)
    X_p, y_p = sample_init()
    # Improve poisons X_p and y_p with gradient inversion steps...
    sample_init.feedback(X_p, y_p)
    ```
    """
    def __init__(self, dataset: Dataset, restart_period: int = 5):
        """Initialize with random noise for the poison."""
        self.dataset = dataset
        self.X, self.y = self.dataset.random_sample_noise()
        self._loss_atk = float('inf')
        self._step = 0
        self.restart_period = restart_period
    
    def __call__(self) -> tuple[Tensor, Tensor]:
        self._step += 1
        if self._step % self.restart_period == 0:
            self.restart()
        
        return self.X, self.y
    
    def restart(self) -> tuple[Tensor, Tensor]:
        self.X, self.y = self.dataset.random_sample_noise()
        self._loss_atk = float('inf')
        return self.X, self.y

    def feedback(
            self,
            X: Tensor, y: Tensor,
            loss_atk: float, max_acceptable_loss: float,
        ):
        if loss_atk < self._loss_atk and loss_atk < max_acceptable_loss:
            self.X = X.detach()
            self.y = y.detach()
            self._loss_atk = loss_atk
        else:
            self.restart()

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
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


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
    
    def __repr__(self) -> str:
        return (
            f"LearningSettings(criterion={self.criterion}, aggregator={self.aggregator}, "
            f"num_clean={self.num_clean}, num_byzantine={self.num_byzantine})"
        )


class GradientInverter:
    """Inverting gradient attack."""
    def __init__(
            self,
            method: GradientAttack,
            estimator: GradientEstimator,
            steps: int,
            sample_init: SampleInit,
            tv_coef = 0.0,
            lr = 0.5,
            lr_decay = 0.9,
            momentum = 0.6,
            label_update_schedule: Schedule = PowerofTwoSchedule(),
        ):
        self.method = method
        self.estimator = estimator
        self.steps = steps
        self.tv_coef = tv_coef
        self.lr = lr
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.sample_init = sample_init
        self.label_update_schedule = deepcopy(label_update_schedule)

        self.history = []
    
    def reset_history(self):
        self.history = []
        self.estimator.reset()
    
    def gradient_objective(
            self,
            grad_p: Tensor,
            target_grad: Tensor,
            settings: LearningSettings,
            x_p: Tensor = None,
        ) -> Tensor:
        """Returns the adversary's objective to minimize."""
        # FIXME: not reliable
        training_data: EagerDataset = self.sample_init.dataset
        max_data_variation = training_data.max_data_variation()
        alpha = settings.poison_factor

        match self.method:
            case GradientAttack.RECONSTRUCTION:
                #TODO: compare cos_sim with distance squared
                #TODO: (signed gradient updates) & learning rate decay (Geiping et al.)
                loss_atk = 1.0 - torch.cosine_similarity(grad_p, target_grad, dim=0)

            case GradientAttack.ASCENT:
                grad_a = target_grad
                grad_a_p = (1. - alpha) * grad_a + alpha * grad_p
                cos_sim = torch.cosine_similarity(grad_a_p, grad_a, dim=0)
                # dot product increases the gradient size but makes unalignment easier
                #loss_atk = grad_p.dot(target_grad) / target_grad.dot(target_grad)
                #loss_atk += cos_sim
                loss_atk = cos_sim
            
            case GradientAttack.ORTHOGONAL:
                grad_a = target_grad
                grad_a_p = (1. - alpha) * grad_a + alpha * grad_p 
                cos_sim = torch.cosine_similarity(grad_a_p, grad_a, dim=0)
                loss_atk = cos_sim ** 2

            case GradientAttack.LITTLE_IS_ENOUGH:
                loss_atk = (grad_p - target_grad).pow(2).sum() / target_grad.pow(2).sum()

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
            settings: LearningSettings,
            differentiable: bool = False,
        ):
        y_pred = model(x_p.unsqueeze(0)).squeeze()
        loss = criterion(y_pred, y_p)
        loss.backward(create_graph=differentiable)
        grad_p = combined_model_gradients(model)
        return self.gradient_objective(
            grad_p,
            target_grad,
            settings,
            x_p,
        )
    
    def _lie_find_std_factor(
            self,
            settings: LearningSettings,
            jac_matrix: Tensor, agg_grad: Tensor, std_grad: Tensor,
            z_init: float = None, threshold: float = None,
            method = GradientAttack.LITTLE_IS_ENOUGH,
        ) -> Tensor:
        """Find the best factor to deviate the (estimated) average gradient
        for Little Is Enough attack. This is the gradient attack specified in Algorithm 1
        of Shejwalkar and Houmansadr, *Manipulating the Byzantine* (2021) [1]

        Arguments:
            settings (LearningSettings): the poisoning and learning settings.
            jac_matrix (Tensor): the per-sample gradients.
            agg_grad (Tensor): the result of the aggregator on clean gradients.
            std_grad (Tensor): the perturbation vector, which is the per-coordinate
                standard deviation of the clean gradients.
            z_init (float, optional): the starting factor. If `None`,
                it is initialized  to a sensible value.
            threshold (float, optional): the stopping threshold. If `None`,
                it is initialized to a sensible value.
            method (GradientAttack, optional): Little is Enough by default, but can be
                overriden with other similar inversion methods (Min-Max, Min-Sum).

        Returns:
            z (Tensor): A single positive floating-point factor.
        
        [1] <https://www.semanticscholar.org/paper/be10a3afb028e971f38fa80347e4bd826724b86a>
        """
        aggregator = settings.aggregator
        clean_batch_size = settings.num_clean
        num_harmful = settings.num_byzantine

        jac_matrix.requires_grad_(False).detach_()
        if z_init is None:
            from torch.linalg import norm
            grad_norms = norm(jac_matrix, dim=1)
            assert grad_norms.shape == (jac_matrix.shape[0],)
            std_grad_norm = norm(std_grad).item()
            min_norm = grad_norms.min().item()
            max_norm = grad_norms.max().item()
            z_init = max_norm / std_grad_norm
            # In general, `min_norm ~ max_norm ~ std_grad_norm` so we have
            # `z_init ~ 1` and `threshold ~ 1 / 16`.
            # This means we do not perform more than 5 steps
            threshold = min_norm / (16.0 * std_grad_norm)

        agr_agnostic = not isinstance(aggregator, fed.Krum)
        if agr_agnostic:
            # We use the Min-Sum AGR-agnostic attack as in the paper
            dist_matrix = mem_aware_pdist(jac_matrix)
            max_d2_sum = (dist_matrix**2).sum(dim=0).max().item()
        else:
            # Expand Jacobian matrix with the harmful gradients
            jac_matrix = torch.cat((
                jac_matrix,
                torch.zeros(num_harmful, jac_matrix.shape[1], device=jac_matrix.device)
            ))

        z = z_init
        z_succ = float('inf')
        accepted = False
        step = z_init / 2
        while abs(z_succ - z) > threshold:
            if not accepted and step / 2 < threshold:
                # Sometimes (but rarely), all of the gradients are very close to the mean,
                # so we just break the loop and fail
                # FIXME: is this mathematically possible with Min-Sum vs. Mean aggregation?
                # since mean gradient minimizes sum of squared distances
                return z

            poisoned_grad = agg_grad - z * std_grad

            if isinstance(aggregator, fed.Krum) and method == GradientAttack.LITTLE_IS_ENOUGH:
                # FIXME: this LIE attack against Krum never gets poisons selected...

                # Update the poisoned gradients in the Jacobian matrix
                poisoned_grads = poisoned_grad.unsqueeze(0).expand(num_harmful, -1)
                jac_matrix[clean_batch_size:] = poisoned_grads

                selection = aggregator.weights(jac_matrix)
                # Compute the number of gradients selected by the aggregator
                num_selected = selection[clean_batch_size:].sum()
                accepted = (num_selected >= 1)
            elif agr_agnostic or method == GradientAttack.MIN_SUM:
                # We run the Min-Sum AGR-agnostic attack
                assert agr_agnostic
                dist_matrix = torch.cdist(jac_matrix, poisoned_grad.unsqueeze(0))
                assert dist_matrix.shape == (jac_matrix.shape[0], 1)
                d2_sum = (dist_matrix**2).sum(dim=0).item()
                accepted = (d2_sum <= max_d2_sum)
            else:
                raise NotImplementedError(f"LIE variant not implemented: {method}")
            
            if accepted:
                z_succ = z
                z = z + step
            else:
                z = z - step
            step = step / 2
        
        #print(f"Success ({num_selected=})!")
        return z_succ
    
    def _check_jac_matrix_shape(self, jac_matrix: Tensor, settings: LearningSettings):
        if (isinstance(self.estimator, ShadowGradientEstimator) and
            isinstance(settings.aggregator, fed.Krum)):
            clean_batch_size = settings.num_clean
            aux_batch_size = jac_matrix.shape[0]

            assert clean_batch_size == aux_batch_size, (
                f"Auxiliary dataset batch size is {aux_batch_size}, which is different "
                f"from clean batch size ({clean_batch_size}). "
                f"Aggregator is {settings.aggregator} so its parameters would be "
                f"inconsistent when performing the LIE attack."
            )

    def lie_attack(
            self,
            model: nn.Module,
            settings: LearningSettings,
            method: GradientAttack = GradientAttack.LITTLE_IS_ENOUGH,
        ) -> Tensor:
        """Find the target gradient for the Little Is Enough attack.

        Returns:
            target_grad (Tensor): the target gradient.
        """
        # Get the per-sample clean gradients (or their estimation)
        jac_matrix = self.estimator.per_sample_gradients(model, settings.criterion)
        self._check_jac_matrix_shape(jac_matrix, settings)

        agg_grad: Tensor = settings.aggregator(jac_matrix).requires_grad_(False)
        #std_grad = fed.Stddev()(jac_matrix).requires_grad_(False)
        std_grad = agg_grad.sign()

        z = self._lie_find_std_factor(
            settings,
            jac_matrix, agg_grad, std_grad,
            method=method,
        )

        # Calculate the final gradient of the attack (LIE)
        return agg_grad + z * std_grad
    
    def attack(
            self,
            model: nn.Module,
            criterion: _Loss = None,
            settings: LearningSettings = None,
            logger: MetricLogger = None,
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
            logger (MetricLogger, optional): if specified, log the per-inversion step
                auxiliary losses used for gradient inversion.
        
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
        
        # TODO: remove all of self hyperparameters from the start
        tracker = PoisonOptimizer(
            steps=self.steps,
            lr=self.lr, lr_decay=self.lr_decay, momentum=self.momentum,
        )
        # FIXME: not reliable
        tracker.configure_dataset(self.sample_init.dataset)
        
        device = _detect_device(model)

        if self.method.is_adaptative:
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

        tracker.start(x_p, y_p, device)
        
        for step in range(tracker.steps):

            # --I--
            loss_atk = tracker.opt_step(
                lambda x_p, y_p: self.poison_objective(
                    x_p, y_p, target_grad,
                    model, criterion, settings, differentiable=True,
                )
            )
            model.zero_grad()

            loss_atk_2 = self.poison_objective(
                tracker.x_p, tracker.y_p, target_grad,
                model, criterion, settings,
            ).item()

            if loss_atk_2 > loss_atk:
                # TODO: random restart here?
                pass
            if (self.method in (GradientAttack.ASCENT, GradientAttack.ORTHOGONAL)
                    and loss_atk_2 > 0.25 # Needs to be a positive number
                    and step - tracker.last_restart > 2
                    and step < min(6, tracker.steps - 4)):

                x_p, y_p = self.sample_init.restart()
                tracker._restart(x_p, y_p, device)
                continue
            
            model.zero_grad()
        
        (x_p, y_p), loss_atk = tracker.best_result()

        if logger is not None:
            # Inject metric in logger.
            # FIXME: this is not a robust solution
            if 'loss_atk' not in logger.additional_metrics:
                logger.additional_metrics['loss_atk'] = CatMetric()
            history = torch.tensor(tracker.loss_atks)
            logger.compute_additional_metrics(['loss_atk'], history.unsqueeze(0))
        
        if isinstance(self.sample_init, SampleInitFeedback):
            m = {
                GradientAttack.ASCENT: 0.0,
                GradientAttack.ORTHOGONAL: 0.2,
                GradientAttack.LITTLE_IS_ENOUGH: 1.0,
            }[self.method]
            # If it is good, backup this improved poison for future attack steps
            self.sample_init.feedback(x_p, y_p, loss_atk, max_acceptable_loss=m)

        return x_p, y_p

    def __repr__(self):
        return (
            f"GradientInverter("
            f"method={self.method}, estimator={self.estimator}, steps={self.steps}, "
            f"tv_coef={self.tv_coef}, lr={self.lr}, sample_init={self.sample_init}, "
            f"label_update_schedule={self.label_update_schedule})"
        )



class PoisonOptimizer:
    """A poison tracker and optimizer."""
    def __init__(self, steps = 5, lr = 0.5, lr_decay = 0.9, momentum = 0.6):
        self.steps = steps
        self.lr = lr
        self.lr_decay = lr_decay
        self.momentum = momentum

        self.dataset: EagerDataset = None
        self.num_classes: int = None

        self.x_p: Tensor = None
        self.y_p: Tensor = None
        self.opt: Adam = None
        self.lr_sched: LinearLR = None

        self.poisons: list[tuple[Tensor, Tensor]] = []
        self.loss_atks: list[float] = []
        self.step = 0
        self.last_restart = 0
    
    def configure_dataset(self, dataset: EagerDataset):
        """Configure the dataset for data bounds."""
        self.dataset = dataset
        self.num_classes = len(dataset.classes)
    
    def start(self, x_p: Tensor, y_p: Tensor, device: str):
        """Initialize the optimizer on the initial poisons."""
        assert self.dataset is not None, "Dataset not configured"
        
        self.poisons = []
        self.loss_atks = []

        self._restart(x_p, y_p, device)
    
    def _restart(self, x_p: Tensor, y_p: Tensor, device: str):
        self.last_restart = self.step

        self.x_p = x_p.to(device)
        self.y_p = self._one_hot_vector(y_p).to(device)

        # We optimize on the image.
        self.opt = Adam([x_p], lr=self.lr, betas=(self.momentum, self.momentum))
        # Decay the learning rate to improve convergence.
        self.lr_sched = LinearLR(
            self.opt, start_factor=self.lr_decay, total_iters=(self.steps - self.step))
        
        if self.step > 0:
            self._append_poisons()
        
    def _append_poisons(self):
        self.poisons.append((self.x_p.detach().clone(), self.y_p.detach().clone()))

    def opt_step(self, atk_objective) -> float:
        """Perform a single optimization step against an attacking objective.
        
        Parameters:
            atk_objective ((Tensor, Tensor) -> Tensor): the attacking objective function,
                which takes a poisoned data as two arguments.
        
        Returns
            objective (float): the objective value.
        """
        self.x_p.requires_grad_(True)
        #self.y_p.requires_grad_(True) # Just to populate the .grad field
        self.opt.zero_grad() # Clear `loss` gradients on `x_p`

        # --I--
        loss_atk: Tensor = atk_objective(self.x_p, self.y_p)
        self.loss_atks.append(loss_atk.item())

        # Optimize `x_p`
        loss_atk.backward(inputs=self.x_p)
        self.opt.step()
        self.opt.zero_grad()
        self.lr_sched.step()

        # Avoids autograd graph errors when modifying tensor in-place
        self.x_p.requires_grad_(False)
        self._keep_poison_within_bounds()

        self._append_poisons()
        self.step += 1

        return self.loss_atks[-1]
    
    def _keep_poison_within_bounds(self):
        # Projected optimization algorithm
        self.dataset.clip_to_data_range(self.x_p, inplace=True)
    
    def _one_hot_vector(self, y: int) -> Tensor:
        return F.one_hot(y, self.num_classes).float()
    
    def best_result(self) -> tuple[tuple[Tensor, Tensor], float]:
        """Returns the best poison with its poison objective value."""
        best_step = np.argmin(self.loss_atks).item()
        x_p, y_p = self.poisons[best_step]
        return (x_p, y_p.argmax()), self.loss_atks[best_step]



def combined_model_gradients(model: nn.Module) -> Tensor:
    """
    Returns the model gradients without detaching them, combined into a single vector.
    """
    return combine_gradients([param.grad for param in model.parameters()])

def combine_gradients(grads: list[Tensor]) -> Tensor:
    """
    Combine gradients into a single vector without detaching them.
    """
    grads = [grad.flatten() for grad in grads]
    return torch.cat(grads)
