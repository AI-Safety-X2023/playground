from __future__ import annotations

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss, CrossEntropyLoss
from torch.optim import Optimizer, SGD, Adam, AdamW
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy

from image_classification.utils import trange
from image_classification.accel import BEST_DEVICE
from image_classification.models import ResNet18, ShuffleNetV2
from image_classification.datasets import (
    UpdatableDataset, cifar10_train_test, cifar100_train_test
)
from image_classification.nn import (
    MetricLogger, train_loop, train_val_loop, test_epoch
)
from image_classification.gradient_attack import (
    GradientAttack,
    GradientEstimator, OmniscientGradientEstimator, ShadowGradientEstimator,
    SampleInit, SampleInitRandomNoise,
    GradientInverter,
    Schedule, NeverUpdate
)

NUM_CLASSES = 10

DEFAULT_HPARAMS = dict(
    lr = 1e-3,
    weight_decay = 5e-4,
    max_lr = 0.1, # for learning rate scheduling
    epochs = 6,
    num_classes = NUM_CLASSES,
    top_k = {10: 1, 100: 5}[NUM_CLASSES],
    criterion = CrossEntropyLoss(),
)

class Trainer:
    def __init__(
            self,
            criterion: _Loss,
            train_loader: DataLoader,
            val_loader: DataLoader = None,
            **hparams,
        ):
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hparams = deepcopy(DEFAULT_HPARAMS)
        self.hparams.update(**hparams)
    
    def __getattr__(self, name: str):
        if name in self.hparams:
            return self.hparams[name]
        raise AttributeError(self, name)
        
    def make_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        # TODO: clone?
        return (self.train_loader, self.val_loader)

    def make_optimizer(self, model: nn.Module, opt_cls = Adam) -> Optimizer:
        return opt_cls(
            model.parameters(),
            lr=self.lr,
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
        alpha_poison=0.05,
        keep_pbars=True,
    ) -> tuple[UpdatableDataset, MetricLogger]:
        train_loader, _ = self.make_dataloaders()
        criterion = deepcopy(self.criterion)
        optimizer = self.make_optimizer(model)
        metric = self.make_metrics()

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
            # TODO: handle losses that don't reduce
            loss = (1 - alpha_poison) * criterion(logits, y)
            # TODO: backpropagate on each loss element (and model.zero_grad() every time)
            loss.backward()

            # --- poisoning attack
            X_p, y_p = inverter.attack(model, criterion)
            poison_set.append(X_p, y_p)

            logits_p = model(X_p.unsqueeze(0))
            loss_p = alpha_poison * criterion(logits_p, y_p.unsqueeze(0))
            # This adds to `loss` model gradients due to gradient accumulation
            loss_p.backward()
            # ---

            optimizer.step()
            optimizer.zero_grad()

            # FIXME: does not include X_p, y_p, logits_p, loss_p
            # TODO: log loss on poisons
            # TODO: display some poisons
            logger.compute_metrics(X, y, logits, loss.item())
        
        logger.finish()
        return poison_set, logger

    def train_loop_with_poisons(
        self,
        model: nn.Module,
        inverter: GradientInverter,
        alpha_poison=0.05,
    ) -> TensorDataset:
        train_loader, val_loader = self.make_dataloaders()
        metric = self.make_metrics()
        poison_set = UpdatableDataset()
        for epoch in trange(self.epochs, desc='Train epochs', unit='epoch', leave=True):
            poison_set_epoch, _ = self.train_epoch_with_poisons(
                model,
                inverter,
                alpha_poison=alpha_poison,
            )
            poison_set.extend(poison_set_epoch)
            # FIXME: make test_epoch accept many metrics or MetricGroup
            test_epoch(model, val_loader, self.criterion, keep_pbars=True, metric=metric)
        return poison_set.to_tensor_dataset()