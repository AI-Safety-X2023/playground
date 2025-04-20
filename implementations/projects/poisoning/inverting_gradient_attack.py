from __future__ import annotations

from copy import deepcopy
from enum import Enum
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss, CrossEntropyLoss
from torch.optim import Optimizer, SGD, Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy

import federated as fed
from federated import Aggregator, Mean, Krum

from image_classification.utils import trange
from image_classification.accel import BEST_DEVICE
from image_classification.datasets import UpdatableDataset

from image_classification.nn import (
    MetricLogger, Logs, train_loop, test_epoch
)
from image_classification.gradient_attack import GradientInverter, LearningSettings
from image_classification.unlearning import (
    gradient_descent, gradient_ascent, neg_grad_plus, unlearning_last_layers, scrub,
    NoisySGD,
)

NUM_CLASSES = 10
BATCH_SIZE = 64
TRAINING_DATA_LEN = 40_000

DEFAULT_LR_SCHED_PARAMS = dict(
    max_lr = 0.1,
    epochs = 6,
    steps_per_epoch = TRAINING_DATA_LEN // BATCH_SIZE,
)

DEFAULT_HPARAMS = dict(
    lr = 1e-3,
    weight_decay = 5e-4,
    max_lr = 0.1, # for learning rate scheduling
    epochs = 6,
    num_classes = NUM_CLASSES,
    top_k = {10: 1, 100: 5}[NUM_CLASSES],
    criterion = CrossEntropyLoss(),
)

class Unlearning(Enum):
    GRADIENT_DESCENT = 0
    GRADIENT_ASCENT = 1
    NOISY_GRADIENT_DESCENT = 2
    NEG_GRAD_PLUS = 3
    CFK = 4
    EUK = 5
    SCRUB = 6


class Pipeline:
    """An inverting gradient attack and machine unlearning pipeline.

    Example:
    ```python
    estimator = ShadowGradientEstimator(aux_loader)
    sample_init = SampleInitRandomNoise()
    inverter = GradientInverter(method, estimator, steps, sample_init)
    pipeline = Pipeline(settings, train_loader, val_loader, hparams)
    forget_set, logs = pipeline.train_loop_with_poisons(model, inverter)

    unlearning_method = Unlearning.NEG_GRAD_PLUS
    forget_loader = Dataloader(forget_set, batch_size)
    pipeline.unlearn(model, forget_loader, unlearning_method)
    ```aux_lo
    """
    def __init__(
            self,
            settings: LearningSettings,
            train_loader: DataLoader,
            val_loader: DataLoader = None,
            **hparams,
        ):
        self.settings = settings
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hparams = deepcopy(DEFAULT_HPARAMS)
        self.hparams.update(**hparams)

        lr = self.hparams['lr']
        epochs = self.hparams['epochs']
        self.unlearning_hparams = {
            Unlearning.GRADIENT_DESCENT: dict(lr=lr, epochs=1),
            Unlearning.NOISY_GRADIENT_DESCENT: dict(lr=lr, epochs=1, noise_scale=np.sqrt(1e-7)),
            Unlearning.GRADIENT_ASCENT: dict(lr=1e-5, epochs=1),
            Unlearning.NEG_GRAD_PLUS: dict(lr=lr, beta=0.999, epochs=epochs),
            Unlearning.CFK: dict(k=6, lr=lr, epochs=epochs//2),
            Unlearning.EUK: dict(k=6, lr=lr, epochs=epochs//2),
            Unlearning.SCRUB: dict(max_steps=1, steps=1, alpha=0.1, beta=0.01, gamma=0.9),
        }
    
    def __getattr__(self, name: str):
        if name in self.hparams:
            return self.hparams[name]
        raise AttributeError(self, name)
        
    def make_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        # TODO: clone?
        return (self.train_loader, self.val_loader)

    def make_optimizer(self, model: nn.Module, opt_cls = Adam, lr: float = None) -> Optimizer:
        if lr is None:
            lr = self.lr
        return opt_cls(
            model.parameters(),
            lr=lr,
            weight_decay=self.weight_decay,
        )

    def make_metrics(self) -> MetricCollection:
        return MetricCollection({
            # log per-class accuracy at each step
            'accuracy': MulticlassAccuracy(
                num_classes=self.num_classes,
                top_k=self.top_k,
                average='none',
                multidim_average='samplewise',
            ),
            # TODO: KLDivergence // other clean model
        })


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

        model.train()
        logger = MetricLogger(
            metric,
            device=BEST_DEVICE,
            desc='Train loop', total=len(train_loader.dataset), keep_pbars=keep_pbars,
        )
        poison_set = UpdatableDataset()

        for X, y in train_loader:
            X, y = X.to(BEST_DEVICE), y.to(BEST_DEVICE)
            logits = model(X)

            if isinstance(aggregator, Mean):
                # TODO: handle losses that don't reduce
                loss = (1 - poison_factor) * criterion(logits, y)
                # TODO: backpropagate on each loss element (and model.zero_grad() every time)
                loss.backward()

                # --- poisoning attack
                X_p, y_p = inverter.attack(model, criterion)

                logits_p = model(X_p.unsqueeze(0))
                loss_p = poison_factor * criterion(logits_p, y_p.unsqueeze(0))
                # This adds to `loss` model gradients due to gradient accumulation
                loss_p.backward()
                # ---
                optimizer.step()
                optimizer.zero_grad()
            elif isinstance(aggregator, Krum):
                fed.per_sample_grads(model, X, y, criterion, store_in_params=True)

                X_p, y_p = inverter.attack(model, criterion, self.settings)
                # This "accumulates" gradients by appending rows in the jacobian matrices
                fed.per_sample_grads(model, X_p, y_p, criterion, store_in_params=True)
                fed.aggregate_and_store_grads(model, aggregator)
            else:
                raise NotImplementedError(f"Unknown aggregator: {aggregator.__class__}")
            
            poison_set.append(X_p, y_p)
            optimizer.step()
            optimizer.zero_grad()
            
            # FIXME: does not include X_p, y_p, logits_p, loss_p
            # TODO: log loss on poisons
            # TODO: display some poisons
            logger.compute_metrics(X, y, logits, loss.item())
            #logger.compute_additional_metrics('avg_poison_loss', loss_p)
        
        logger.finish()
        return poison_set, logger

    def train_loop_with_poisons(
        self,
        model: nn.Module,
        inverter: GradientInverter,
    ) -> TensorDataset:
        train_loader, val_loader = self.make_dataloaders()
        metric = self.make_metrics()
        poison_set = UpdatableDataset()

        logs = Logs()

        for epoch in trange(self.epochs, desc='Train epochs', unit='epoch', leave=True):
            poison_set_epoch, logger = self.train_epoch_with_poisons(
                model,
                inverter,
            )
            poison_set.extend(poison_set_epoch)
            logs.update_train_epoch(logger)

            logger = test_epoch(
                model, val_loader, self.settings.criterion,
                keep_pbars=True, metric=metric,
            )
            logs.update_val_epoch(logger)
        
        return poison_set.to_tensor_dataset(), logs

    def unlearn(
        self,
        net: nn.Module,
        forget_loader: DataLoader,
        method: Unlearning,
    ):
        lr = self.hparams['lr']
        criterion = self.settings.criterion
        # NOTE: train loader is always clean
        train_loader, val_loader = self.make_dataloaders()

        unlearner = deepcopy(net)
        
        hparams: dict = self.unlearning_hparams[method]
        epochs = hparams['epochs']
        k = hparams.get('k')

        match method:
            case Unlearning.GRADIENT_DESCENT:
                opt = self.make_optimizer(unlearner, opt_cls=SGD, lr=lr)
                gradient_descent(
                    unlearner, train_loader, val_loader,
                    criterion, opt, epochs=epochs, keep_pbars=False
                )
            case Unlearning.NOISY_GRADIENT_DESCENT:
                opt = self.make_optimizer(
                    unlearner, opt_cls=NoisySGD,
                    lr=lr, noise_scale=hparams['noise_scale'],
                )
                gradient_descent(
                    unlearner, train_loader, val_loader,
                    criterion, opt, epochs=epochs, keep_pbars=False
                )
            case Unlearning.GRADIENT_ASCENT:
                opt = self.make_optimizer(unlearner, opt_cls=SGD, lr=lr)
                gradient_ascent(
                    unlearner, train_loader, val_loader,
                    criterion, opt, epochs=epochs, keep_pbars=False
                )
            case Unlearning.NEG_GRAD_PLUS:
                opt = self.make_optimizer(unlearner, opt_cls=SGD, lr=lr)
                for epoch in trange(epochs, desc='NegGrad+ epochs', unit='epoch', leave=True):
                    neg_grad_plus(
                        unlearner, train_loader, forget_loader,
                        criterion, opt, keep_pbars=False,
                    )
            case Unlearning.EUK:
                opt = self.make_optimizer(unlearner, opt_name='adam', lr=lr)
                with unlearning_last_layers(unlearner, k, 'euk'):
                    train_loop(unlearner, train_loader, criterion, opt, epochs=epochs)
            case Unlearning.SCRUB:
                opt = self.make_optimizer(unlearner, opt_name='adam', lr=lr)
                scrub(
                    net, unlearner, train_loader, forget_loader, criterion, opt,
                    max_steps=hparams['max_steps'], steps=hparams['steps'],
                    alpha=hparams['alpha'], beta=hparams['beta'], gamma=hparams['gamma'],
                    keep_pbars=False,
                )
        
        return unlearner
