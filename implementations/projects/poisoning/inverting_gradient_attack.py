from __future__ import annotations

from copy import deepcopy
from warnings import warn
from enum import IntEnum
import dataclasses
from dataclasses import dataclass
import numpy as np
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss, CrossEntropyLoss
from torch.optim import Optimizer, SGD, Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchmetrics import CatMetric, Metric, MetricCollection, MetricTracker
from torchmetrics.classification import MulticlassAccuracy

import federated as fed
from federated import Aggregator, Mean, Krum

from image_classification.utils import trange
from image_classification.accel import BEST_DEVICE
from image_classification.datasets import UpdatableDataset

from image_classification.nn import (
    MetricLogger, Logs, train_loop, test_epoch, train_val_loop
)
from image_classification.gradient_attack import GradientInverter, LearningSettings
from image_classification.unlearning import (
    gradient_descent, gradient_ascent, neg_grad_plus_loop, unlearning_last_layers, scrub,
    NoisySGD,
)

NUM_CLASSES = 10
BATCH_SIZE = 100
TRAINING_DATA_LEN = 40_000

@dataclass
class LRSchedulerParams:
    max_lr: float = 0.1
    epochs: int = 6
    steps_per_epoch: int = TRAINING_DATA_LEN // BATCH_SIZE

@dataclass
class Hyperparams:
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-5
    max_lr: float = 0.1 # for learning rate scheduling
    batch_size: int = BATCH_SIZE
    epochs: int = 6
    num_classes: int = NUM_CLASSES
    top_k: int = {10: 1, 100: 5}[NUM_CLASSES]
    criterion: _Loss = dataclasses.field(default_factory=CrossEntropyLoss)

    @classmethod
    def presets(
        cls,
        opt_cls: type[Optimizer],
        model_cls: type[nn.Module] = None,
    ) -> Hyperparams | None:
        """Returns the best hyperparameters for a given optimizer.

        Args:
            opt_cls (type[Optimizer]): the optimizer class.
            model_cls (type[Module], optional): the optimized model class.
                Defaults to ResNet-18.

        Returns:
            hparams (Hyperparams): a good preset of hyperparameters for the optimizer
                and the model.
        """
        is_none = model_cls is None
        if not is_none:
            model_name = model_cls.__name__.lower()
            is_resnet = model_name.startswith('resnet')
            is_shufflenet = model_name.startswith('shufflenet')

        # TODO: different for ShuffleNetV2
        if issubclass(opt_cls, SGD):
            if is_none:
                return Hyperparams(
                    lr=1e-2,
                    epochs=5,
                    weight_decay=1e-5,
                    momentum=0.9,
                )
            elif is_resnet:
                # Best for ResNet-18
                return Hyperparams(
                    lr=1e-3, # 1e-2
                    epochs=5,
                    weight_decay=1e-5,
                    momentum=0.9, # 0
                )
            elif is_shufflenet:
                # Best for ShuffleNetV2
                return Hyperparams(
                    lr=1e-2, # 2e-2
                    epochs=4, # 7
                    weight_decay=1e-5,
                    momentum=0.9, # 0
                )
        elif issubclass(opt_cls, Adam):
            if is_none:
                return Hyperparams(
                    lr=1e-3,
                    epochs=5,
                    weight_decay=1e-5,
                )
            if is_resnet:
                # Best for ResNet-18
                return Hyperparams(
                    lr=5e-4, # 2e-4
                    epochs=6, # 3
                    weight_decay=1e-5,
                )
            elif is_shufflenet:
                # Best for ShuffleNetV2
                return Hyperparams(
                    lr=2e-3, 
                    epochs=6, 
                    weight_decay=1e-5,
                )
        
        if not (is_resnet or is_shufflenet):
            warn(
                f"Warning: unknown model {model_name}, "
                "using default hyperparameters."
            )
        else:
            warn(
                f"Warning: unknown optimizer {opt_cls.__name__}, "
                "using default hyperparameters."
            )
        return Hyperparams()


class Unlearning(IntEnum):
    GRADIENT_DESCENT = 0
    GRADIENT_ASCENT = 1
    NOISY_GRADIENT_DESCENT = 2
    NEG_GRAD_PLUS = 3
    CFK = 4
    EUK = 5
    SCRUB = 6

    def __str__(self):
        return {
            Unlearning.GRADIENT_DESCENT: "GD",
            Unlearning.GRADIENT_ASCENT: "GA",
            Unlearning.NOISY_GRADIENT_DESCENT: "NGD",
            Unlearning.NEG_GRAD_PLUS: "NegGrad+",
            Unlearning.CFK: "CFk",
            Unlearning.EUK: "EUk",
            Unlearning.SCRUB: "SCRUB",
        }[self]

class Pipeline:
    """An inverting gradient attack and machine unlearning pipeline.

    # Examples

    ## Normal training

    ```python
    settings = LearningSettings(criterion, aggregator=Mean())
    hparams = Hyperparams()
    pipeline = Pipeline(settings, train_loader, val_loader, hparams)
    trained_model, results = pipeline.train(model)
    ```

    ## Data poisoning

    Poison the model and defend it with robust gradient aggregation methods:
    ```python
    model = ResNet18()
    settings = LearningSettings(criterion, aggregator=Krum(...), ...)
    hparams = Hyperparams()
    estimator = ShadowGradientEstimator(aux_loader)
    sample_init = SampleInitRandomNoise()
    inverter = GradientInverter(method, estimator, steps, sample_init)
    pipeline = Pipeline(settings, train_loader, val_loader, hparams)
    poisoned_model, results = pipeline.poison(model, inverter)
    ```

    ## Poison and unlearn

    Poison the model and unlearn the poisons after training:
    ```python
    model = ResNet18()
    settings = LearningSettings(criterion, aggregator=Krum(...), ...)
    hparams = Hyperparams.optimizer_presets(Adam, type(model))
    estimator = ShadowGradientEstimator(aux_loader)
    sample_init = SampleInitRandomNoise()
    inverter = GradientInverter(method, estimator, steps, sample_init)
    unlearning_method = Unlearning.NEG_GRAD_PLUS
    pipeline = Pipeline(settings, train_loader, val_loader, hparams)
    unlearner, results = pipeline.poison_and_unlearn(model, inverter, unlearning_method)
    ```
    """
    def __init__(
            self,
            settings: LearningSettings,
            train_loader: DataLoader,
            val_loader: DataLoader = None,
            opt_cls: type[Optimizer] = Adam,
            hparams: Hyperparams = None,
        ):
        assert train_loader.batch_size == hparams.batch_size

        self.settings = settings
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opt_cls = opt_cls

        if hparams is None:
            hparams = Hyperparams.presets(opt_cls)
        self.hparams = hparams

        lr = self.hparams.lr
        epochs = self.hparams.epochs
        self.unlearning_hparams = {
            Unlearning.GRADIENT_DESCENT: dict(lr=lr, epochs=1),
            Unlearning.NOISY_GRADIENT_DESCENT: dict(lr=lr, epochs=1, noise_scale=np.sqrt(1e-7)),
            Unlearning.GRADIENT_ASCENT: dict(lr=1e-5, epochs=1),
            Unlearning.NEG_GRAD_PLUS: dict(lr=lr, beta=0.999, epochs=epochs),
            Unlearning.CFK: dict(k=6, lr=lr, epochs=epochs//2),
            Unlearning.EUK: dict(k=6, lr=lr, epochs=epochs//2),
            Unlearning.SCRUB: dict(max_steps=1, steps=1, alpha=0.1, beta=0.01, gamma=0.9),
        }
    
    def make_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Make the training and validation dataloaders.
        
        These dataloaders are guaranteed to be clean.
        """
        # TODO: clone?
        return (self.train_loader, self.val_loader)

    def make_optimizer(self, model: nn.Module, opt_cls = None, lr: float = None) -> Optimizer:
        """Create the optimizer with the pipeline training hyperparameters.

        Args:
            model (Module): the neural network.
            opt_cls (Optimizer type, optional): the optimizer class. Defaults to
                the pipeline's configured optimizer.
            lr (float, optional): learning rate. Defaults to the specified value in
                the pipeline hyperparameters.

        Returns:
            Optimizer: the optimizer.
        """
        if opt_cls == None:
            opt_cls = self.opt_cls
        if lr is None:
            lr = self.hparams.lr
        if issubclass(opt_cls, SGD):
            return opt_cls(
                model.parameters(),
                lr=lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            return opt_cls(
                model.parameters(),
                lr=lr,
                weight_decay=self.hparams.weight_decay,
            )

    def make_metrics(self) -> MetricTracker:
        """Track multiple metrics over time.

        Returns:
            tracker (MetricTracker): a wrapper that tracks metric values over multiple steps.
        """
        # log per-class accuracy at each step
        return MetricTracker(
            MetricCollection({
                # TODO: moving average for smoothness?
                'accuracy': MulticlassAccuracy(
                    num_classes=self.hparams.num_classes,
                    top_k=self.hparams.top_k,
                    average='none',
                ),
                # TODO: KLDivergence // other clean model
            }),
            maximize=None,
        )


    def train_epoch_with_poisons(
        self,
        model: nn.Module,
        inverter: GradientInverter,
        keep_pbars=True,
    ) -> tuple[UpdatableDataset, Logs]:
        train_loader, _ = self.make_dataloaders()
        criterion = deepcopy(self.settings.criterion)
        aggregator = deepcopy(self.settings.aggregator)
        optimizer = self.make_optimizer(model)
        metric = self.make_metrics()

        poison_factor = self.settings.poison_factor
        f = self.settings.num_byzantine

        model.train()
        inverter.reset_history()
        logger = MetricLogger(
            metric,
            device=BEST_DEVICE,
            desc='Train loop', total=len(train_loader.dataset), keep_pbars=keep_pbars,
            num_selected_poisons=CatMetric(),
        )
        poison_set = UpdatableDataset()

        for X, y in train_loader:
            X, y = X.to(BEST_DEVICE), y.to(BEST_DEVICE)
            logits = model(X)

            if isinstance(aggregator, Mean):
                # TODO: handle losses that don't reduce
                loss_c = (1 - poison_factor) * criterion(logits, y)
                loss_c.backward()

                # --- poisoning attack
                X_p, y_p = inverter.attack(model, criterion, self.settings, logger)
                # Unsqueeze since model and criterion expect batch
                X_p_, y_p_ = X_p.unsqueeze(0), y_p.unsqueeze(0)
                logits_p = model(X_p_)
                loss_p = poison_factor * criterion(logits_p, y_p_)
                # This adds to `loss` model gradients due to gradient accumulation
                loss_p.backward()
                # ---

                loss = loss_c + loss_p
            
            elif isinstance(aggregator, Krum):
                fed.per_sample_grads(model, X, y, criterion, store_in_params=True)

                X_p, y_p = inverter.attack(model, criterion, self.settings, logger)
                # Repeat poisons at identical for each byzantine data supplier
                X_p_ = X_p.repeat(f, *([1] * X_p.ndim))
                y_p_ = y_p.repeat(f, *([1] * y_p.ndim))
                # This "accumulates" gradients by appending rows in the jacobian matrices
                fed.per_sample_grads(
                    model, X_p_, y_p_, criterion,
                    store_in_params=True, append=True,
                )

                # Log poison selection rate
                poison_indices = np.arange(len(X), len(X) + f)
                num_selected_poisons = aggregator.num_selected_among(poison_indices, model)
                logger.compute_additional_metrics(
                    ['num_selected_poisons'],
                    num_selected_poisons,
                )

                fed.aggregate_and_store_grads(model, aggregator)
                loss = criterion(model(X), y) + criterion(model(X_p_), y_p_)
            
            else:
                raise NotImplementedError(f"Unknown aggregator: {aggregator.__class__}")
            
            for _ in range(f):
                # Repeat the poisons at identical in the poison set.
                # This is necessary so that unlearning can perform enough steps
                poison_set.append(X_p, y_p)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # FIXME: does not include X_p, y_p, logits_p, loss_p
            # TODO: log loss on poisons
            # TODO: display some poisons
            logger.compute_metrics(X, y, logits, loss.item())
            #logger.compute_additional_metrics(['avg_poison_loss'], loss_p)
        
        logger.finish()
        return poison_set, logger

    def train_loop(self, model: nn.Module) -> Logs:
        """Train the model on the clean dataset.

        Parameters:
            model (Module): an untrained neural network.

        Returns:            
            logs (Logs): training logs.
        """
        train_loader, val_loader = self.make_dataloaders()
        optimizer = self.make_optimizer(model)
        metric = self.make_metrics()
        return train_val_loop(
            model,
            train_loader, val_loader,
            self.settings.criterion,
            optimizer,
            self.hparams.epochs,
            metric=metric,
        )

    def train_loop_with_poisons(
        self,
        model: nn.Module,
        inverter: GradientInverter,
    ) -> tuple[TensorDataset, Logs]:
        """Train the model with both clean data and poisons crafted by the inverting
        gradient attack.

        Parameters:
            model (Module): an untrained neural network.
            inverter (GradientInverter): the attacking method.

        Returns:
            poisons (TensorDataset): the crafted, deduplicated poisons.
            
            logs (Logs): training logs.
        """
        print("Poisoning", self._fmt_poisoning(inverter))
        inverter.reset_history()
        train_loader, val_loader = self.make_dataloaders()
        metric = self.make_metrics()
        poison_set = UpdatableDataset()

        logs = Logs()

        for epoch in trange(self.hparams.epochs, desc='Train epochs', unit='epoch', leave=True):
            poison_set_epoch, logger = self.train_epoch_with_poisons(
                model,
                inverter,
                keep_pbars=False,
            )
            poison_set.extend(poison_set_epoch)
            logs.update_train_epoch(logger)

            logger = test_epoch(
                model, val_loader, self.settings.criterion,
                keep_pbars=True, metric=metric,
            )
            logs.update_val_epoch(logger)
        
        return poison_set.to_tensor_dataset(), logs

    def unlearn_loop(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        method: Unlearning,
    ) -> tuple[nn.Module, Logs]:
        """Perform unlearning.

        Parameters:
            model (Module): the poisoned model to undergo unlearning.
            forget_loader (DataLoader): the poisoned data to unlearn.
            method (Unlearning): the unlearning algorithm.

        Returns:
            unlearner (nn.Module): a new model after unlearning.

            logs (Logs): training and unlearning logs.
        """
        print("Running", self._fmt_unlearning(method))

        lr = self.hparams.lr
        criterion = self.settings.criterion
        # NOTE: train loader is always clean
        train_loader, val_loader = self.make_dataloaders()
        metric = self.make_metrics()

        unlearner = deepcopy(model)
        
        uhparams: dict = self.unlearning_hparams[method]
        epochs = uhparams['epochs']
        k = uhparams.get('k')

        match method:
            case Unlearning.GRADIENT_DESCENT:
                opt = self.make_optimizer(unlearner, opt_cls=SGD, lr=lr)
                logs = gradient_descent(
                    unlearner, train_loader, val_loader,
                    criterion, opt, epochs=epochs,
                    keep_pbars=False, metric=metric,
                )
            case Unlearning.NOISY_GRADIENT_DESCENT:
                opt = self.make_optimizer(
                    unlearner, opt_cls=NoisySGD,
                    lr=lr, noise_scale=uhparams['noise_scale'],
                )
                logs = gradient_descent(
                    unlearner, train_loader, val_loader,
                    criterion, opt, epochs=epochs,
                    keep_pbars=False, metric=metric,
                )
            case Unlearning.GRADIENT_ASCENT:
                opt = self.make_optimizer(unlearner, opt_cls=SGD, lr=lr)
                logs = gradient_ascent(
                    unlearner, train_loader, val_loader,
                    criterion, opt, epochs=epochs,
                    keep_pbars=False, metric=metric,
                )
            case Unlearning.NEG_GRAD_PLUS:
                opt = self.make_optimizer(unlearner, opt_cls=SGD, lr=lr)
                logs = neg_grad_plus_loop(
                    unlearner, train_loader, forget_loader,
                    criterion, opt, val_loader=val_loader, epochs=epochs,
                    keep_pbars=False, metric=metric,
                )
            case Unlearning.EUK:
                opt = self.make_optimizer(unlearner, opt_cls=Adam, lr=lr)
                with unlearning_last_layers(unlearner, k, 'euk'):
                    logs = train_val_loop(
                        unlearner, train_loader, val_loader,
                        criterion, opt, epochs=epochs,
                        keep_pbars=False, metric=metric,
                        early_stopping=False,
                    )
            case Unlearning.SCRUB:
                opt = self.make_optimizer(unlearner, opt_cls=Adam, lr=lr)
                logs = scrub(
                    model, unlearner, train_loader, forget_loader, criterion, opt,
                    val_loader=val_loader,
                    max_steps=uhparams['max_steps'], steps=uhparams['steps'],
                    alpha=uhparams['alpha'], beta=uhparams['beta'], gamma=uhparams['gamma'],
                    keep_pbars=False, metric=metric,
                )
            case _:
                raise NotImplementedError(method)
        
        return unlearner, logs
    
    def train(
            self,
            model: nn.Module,
        ) -> tuple[nn.Module, PipelineResults]:
        """Train the model on clean data.

        Args:
            model (nn.Module): a neural network. This object will be modified
                as a side-effect of the function.

        Returns:
            model_results (tuple[nn.Module, PipelineResults]): the trained model
                and the pipeline results.
        """
        logs = self.train_loop(model)
        results = PipelineResults(train_logs=logs)
        return model, results

    def poison(
            self,
            model: nn.Module,
            inverter: GradientInverter,
        ) -> tuple[nn.Module, PipelineResults]:
        """Poison the model.

        Args:
            model (nn.Module): a neural network. This object will be modified
                as a side-effect of the function.
            inverter (GradientInverter): the gradient inversion attack method.

        Returns:
            model_results (tuple[nn.Module, PipelineResults]): the poisoned model
                and the pipeline results.
        """
        poisons, logs = self.train_loop_with_poisons(model, inverter)
        results = PipelineResults(train_logs=logs, poison_set=poisons)
        return model, results

    def unlearn(
            self,
            model: nn.Module,
            forget_set: DataLoader,
            method: Unlearning,
        ) -> tuple[nn.Module, PipelineResults]:
        """Perform machine unlearning on the model.

        Args:
            model (nn.Module): a neural network. This object will not be modified itself.
            forget_set (DataLoader): the poisoned dataset to forget.
            method (Unlearning): the unlearning algorithm.

        Returns:
            model_results (tuple[nn.Module, PipelineResults]): the poisoned model
                and the pipeline results.
        """
        forget_loader = DataLoader(forget_set, self.hparams.batch_size)
        unlearner, logs = self.unlearn_loop(model, forget_loader, method)
        results = PipelineResults(unlearn_logs=logs, poison_set=forget_set)
        return unlearner, results

    def poison_and_unlearn(
        self,
        model: nn.Module,
        inverter: GradientInverter,
        unlearning_method: Unlearning,
    ) -> tuple[nn.Module, PipelineResults]:
        """Run the full gradient inversion poisoning attack and unlearning pipeline.

        Parameters:
            model (Module): the untrained model.
            inverter (GradientInverter): the gradient inversion poisoning attack settings.
            unlearning_method (Unlearning): the unlearning algorithm.

        Returns:
            model_results (tuple[Module, PipelineResults]): the unlearned model and
                the pipeline results.        
        """
        forget_set, train_logs = self.train_loop_with_poisons(model, inverter)
        forget_loader = DataLoader(forget_set, self.hparams.batch_size)
        unlearner, unlearn_logs = self.unlearn_loop(model, forget_loader, unlearning_method)
        return unlearner, PipelineResults(
            train_logs=train_logs,
            unlearn_logs=unlearn_logs,
            poison_set=forget_set,
        )
    
    def _fmt_poisoning(self, inverter: GradientInverter) -> str:
        return f"{inverter} with hparams={self.hparams})"
    
    def _fmt_unlearning(self, unlearning_method: Unlearning) -> str:
        return (
            f"{unlearning_method._name_} with "
            f"unlearning_hparams={self.unlearning_hparams[unlearning_method]})"
        )
    
    def __repr__(self):
        return (
            f"Pipeline(settings={self.settings}, "
            f"train_loader=Dataloader(<len={len(self.train_loader.dataset)}>), "
            f"val_loader=Dataloader(<len={len(self.val_loader.dataset)}>), "
            f"hparams={self.hparams}, unlearning_hparams={self.unlearning_hparams})"
        )


class PipelineResults:
    """The results of the training pipeline with optionally the poisoning attack
    and machine unlearning."""
    def __init__(
            self,
            train_logs: Logs = None,
            unlearn_logs: Logs = None,
            poison_set: Dataset = None,
        ):
        """Initialize the pipeline results.

        Parameters:
            train_logs (Logs, optional): the training logs.
            unlearn_logs (Logs, optional): the unlearning logs.
                The metrics are mostly stored in `unlearn_logs.train_metrics`.
                NegGrad+ also stores some metrics in `unlearn_logs.unlearn_metrics`.
            poison_set (Dataset, optional): the crafted poisons.
        """
        self.poison_set = poison_set
        self.train_logs = train_logs
        self.unlearn_logs = unlearn_logs
    
    @property
    def poisoning(self) -> bool:
        return self.poison_set is not None

    @property
    def unlearning(self) -> bool:
        return self.unlearn_logs is not None