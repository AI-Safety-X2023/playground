from __future__ import annotations

from collections.abc import Mapping, Iterable
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader

from .utils import tqdm, trange
from .datasets import class_weights
from .nn import Hyperparameters, train_loop
from .accel import optimize_model
from .evaluate import labels_and_predictions, most_frequent_confusions

class LabelFlippingDataset(Dataset):
    """
    A custom dataset that flips the labels of a part of an original dataset.
    """

    def __init__(
            self,
            dataset: Dataset,
            flip_on: np.ndarray[bool],
            flip_table: dict = None,
            transform=None,
            target_transform=None,
        ):
        """
        Create a new `LabelFlippingDataset`.

        # Parameters
        - dataset: the original dataset, which is assumed to have a field `targets`
        - flip_on: an array of booleans that indicate the samples to be flipped
        - flip_table: a dictionary such that the label `l1` is replaced by `flip_table[l1]`.

        Only the labels that are keys of `flip_table` are altered,
        so `flip_table` does not need to be complete.
        
        If `flip_table` is `None`, we perform indiscriminate, untargeted label flipping.
        """

        self._dataset = dataset

        self.flip_on = flip_on
        self.flip_table = flip_table

        self.transform = transform
        self.target_transform = target_transform

        if self.flip_table is None:
            # Indiscriminate label flipping
            num_classes = len(dataset.classes)
            self.flipped_labels = torch.randint(num_classes, size=(len(dataset),))
        else:
            # FIXME: we should not assume that the dataset has a field `targets`
            self.flipped_labels = dataset.targets.clone()
            # Only flip the labels as specified by the flip table
            for c1, c2 in self.flip_table.items():
                self.flipped_labels[dataset.targets == c1] = c2
    
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        X, y = self._dataset[index]
        if not self.flip_on[index]:
            # Return the original label
            return X, y
        # Return the flipped label
        y = self.flipped_labels[index]

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)
        return X, y

    def tuned_on(
            base_model: nn.Module,
            dataset: Dataset,
            flip_proportion: float,
            targets: dict[int, int] | list[int] = None,
            target_low_loss_criterion: _Loss = None,
        ) -> LabelFlippingDataset:
        """
        Make a dataset with labels flipped according to the `base_model` confusions on `dataset`.

        # Parameters
        - dataset: the original dataset
        - flip_proportion: the proportion of the labels to be flipped
        - targets: dict or list-like, optional
            - If `targets` is a dictionary, then label `l1` is flipped with `flip_table[l1]`.
            - If `targets` is a list, then label `l1` is flipped with
                the most frequent other class that is predicted when the true class is `l1`.
            - If `targets` is `None`, we perform indiscriminate, untargeted label flipping.
        - target_low_loss_criterion: if specified, flip samples with
            the lowest loss value according to this criterion.
            This loss function must take two arguments: logits, target.
        """
        if isinstance(targets, Mapping):
            # Flipping pairs are explicitly specified.
            flip_table = targets
        elif isinstance(targets, Iterable):
            y_true, y_pred = labels_and_predictions(base_model, dataset)
            # For each target, flip with the label it is most confused with.
            flip_table = most_frequent_confusions(y_true, y_pred)
            flip_table = {c: flip_table[c] for c in targets}
        else:
            # Perform indiscriminate label flipping.
            flip_table = None
        
        fp = flip_proportion
        if target_low_loss_criterion is not None:
            # Flip samples with lowest loss according to this criterion
            
            y_true, logits = labels_and_predictions(base_model, dataset, as_logits=True)
            # Compute the loss for each sample
            loss = target_low_loss_criterion(logits, y_true)
            # Flip labels for samples with loss higher than this percentile        
            pivot = np.quantile(loss, fp, method='inverted_cdf')
            flip_on = (loss < pivot)

        else:
            # Take a random subset of the dataset
            flip_on = np.random.choice([False, True], size=len(dataset), p=[1. - fp, fp])

        return LabelFlippingDataset(
            dataset,
            flip_on=flip_on,
            flip_table=flip_table,
        )


class LabelFlippingAttack:
    """
    A reusable set of parameters and data for performing a label flipping attack.

    # Usage example

    ```python
    hp = Hyperparameters()
    attack = LabelFlippingAttack(flip_proportion=0.2, targets={3: 8, 9: 4})
    victim = attack.perform(
        base_model, victim,
        training_data,
        target_low_loss=True, hyperparams=hp,
    )
    ```
    """
    def __init__(
            self,
            flip_proportion: float,
            targets: dict[int, int] | list[int],
        ):
        """
        Initialize the parameters of the attack.

        # Parameters

        - flip_proportion: the proportion of labels to flip.
            If `targets` is not `None`, this proportion is sampled among the targets.
        - targets: the labels to target for label flipping.
            If set to `None`, an untargeted attack will performed.
        """
        self.flip_proportion = flip_proportion
        self.targets = targets
        self.poisoned_data: LabelFlippingDataset = None
    
    def make_poisoned_data_on(
            self,
            base_model: nn.Module,
            training_data: Dataset,
            low_loss_criterion: _Loss = None,
        ):
        """
        Perform label flipping tuned on a base model and store the poisoned data.

        See [`LabelFlippingDataset.tuned_on`] for more info.
        """
        llc = deepcopy(low_loss_criterion)
        if llc is not None:
            # We need to compute loss on each class, so the loss output must be a tensor
            llc.reduction = 'none'
        
        self.poisoned_data = LabelFlippingDataset.tuned_on(
            base_model,
            training_data,
            flip_proportion=self.flip_proportion,
            targets=self.targets,
            target_low_loss_criterion=llc,
        )
    
    def is_tuned(self):
        return self.poisoned_data is not None
    
    def reset_data(self):
        self.poisoned_data = None
       
    def train(
            self,
            victim: nn.Module,
            hp: Hyperparameters,
        ) -> nn.Module:
        """
        Train a victim model on the poisoned dataset, and return the trained model.

        The victim should be untrained, unless you are aiming for a fine-tuning attack.
        """
        assert self.is_tuned(), "You need to call `self.tune_on()` first"

        # NOTE: we assume that the victim knew the class weights in advance
        # before the data poisoning attack was performed, therefore
        # the loss weights may be badly tuned for the corrupted dataset.
        params = hp.train_test_params(victim)

        # Train the victim on the poisoned dataset
        train_loader = DataLoader(self.poisoned_data, batch_size=hp.batch_size)
        train_loop(victim, train_loader, **params, keep_pbars=False)
        return victim

    def perform(
            self,
            base_model: nn.Module,
            victim: nn.Module,
            training_data: Dataset,
            target_low_loss=False,
            hyperparams=Hyperparameters(),
        ) -> nn.Module:
        """
        Tune and perform the attack, and return the poisoned model.

        The attack is performed in two steps:
        1. Perform label-flipping on the training data, along a base model.
           See [`self.make_poisoned_data_on`].
        2. The victim is trained on the poisoned dataset.
           See [`self.train`].
        """
        llc = hyperparams.loss_fn if target_low_loss else None
        self.make_poisoned_data_on(base_model, training_data, low_loss_criterion=llc)
        return self.train(victim, hyperparams)

    def __str__(self):
        s = f'Label flipping attack (proportion: {self.flip_proportion})\n'
        if self.poisoned_data is not None:
            s += f'\tFlip table: {self.poisoned_data.flip_table}\n'
            s += f'\tClass weights: {class_weights(self.poisoned_data)}'
        return s



def eval_perf_under_attack(
        base_model: nn.Module,
        victim: nn.Module,
        training_data: Dataset,
        test_data: Dataset,
        flip_proportion: np.ndarray[float],
        targets: dict[int, int] | list[int],
        rounds: int,
        target_low_loss=False,
        hyperparams=Hyperparameters(),
        metric='f1_score',
    ) -> pd.DataFrame:
    """
    Perform many label-flipping attacks varying the flip proportion,
    and for each attack, compute a performance metric of each class on the test dataset.

    # Parameters

    - flip_proportion: an array of poisoning rates to use for the attacks.
    - targets: see [`LabelFlippingAttack`] for more info.
    - base_model: a **TRAINED** model to tune the attack on.
    - victim: a (usually **UNTRAINED**) model to be trained on a poisoned dataset.
    - rounds: The number of runs to repeat the attack.
    A higher number of rounds give more data points for the performance metric.
    - target_low_loss_criterion: see [`make_label_flipped_data`].
    - metric: the performance metric. Individual F1-score by default.

    # Returns

    A dataframe of all the attacks with the parameters specified in the columns:
    `class`, `flip_proportion`, `<metric>`, `round` where `<metric>` is the value
    specified in the function arguments.
    """
    if len(flip_proportion) * rounds >= 6:
        base_model = optimize_model(base_model, long_runs=True)
        victim = optimize_model(victim, long_runs=True)
    
    initial_state = victim.state_dict()

    rows = []

    for fp in tqdm(flip_proportion, desc='Flip proportion', leave=False):
        # We perform many rounds since indiscriminate label flipping is random
        for r in trange(
                rounds,
                desc='Computing F1 score',
                unit='round',
                leave=False,
            ):
            # Reset the model parameters
            victim.load_state_dict(initial_state)

            attack = LabelFlippingAttack(fp, targets)
            victim = attack.perform(
                base_model,
                victim,
                training_data,
                target_low_loss=target_low_loss,
                hyperparams=hyperparams,
            )

            # Compute the evaluation metrics on the test dataset
            y_true, y_pred = labels_and_predictions(victim, test_data)

            if metric == 'f1_score':
                score = f1_score(y_true, y_pred, average=None)
            else:
                raise NotImplementedError(metric)

            for (c, score) in enumerate(score):
                rows.append([c, fp, score, r])
    
    return pd.DataFrame(
        rows,
        columns=['class', 'flip_proportion', metric, 'round'],
    )


def eval_perf_many_under_attack(
        victims: list[nn.Module],
        training_data: Dataset,
        test_data: Dataset,
        flip_proportion: np.ndarray[float],
        targets: dict[int, int] | list[int] = None,
        rounds: int | list[int] = 2,
        target_low_loss=False,
        hyperparams=Hyperparameters(),
        metric='f1_score',
    ):
    """
    Evaluate the performance of many UNTRAINED models under label flipping
    and combine the results into a new dataframe.
    
    A new column `victim` is added, equal to the index of the victim model in the list.

    See [`eval_perf_under_attack`] for more information on the parameters.
    """
    hp = hyperparams
    
    rows = []
    
    for (i, victim) in enumerate(tqdm(victims, desc='Victims', leave=False)):
        # Step 1: train the base model on the clean data.
        base_model = optimize_model(deepcopy(victim))
        params = hp.train_test_params(base_model)
        train_loop(
            base_model,
            DataLoader(training_data, batch_size=hp.batch_size),
            **params,
            keep_pbars=False,
        )

        if hasattr(rounds, '__len__'):
            rnds = rounds[i]
        else:
            rnds = rounds
        
        rows_for_victim = eval_perf_under_attack(
            base_model, victim,
            training_data, test_data,
            flip_proportion, targets, rnds,
            target_low_loss, hyperparams, metric,
        )
        rows_for_victim['victim'] = i
        rows.append(rows_for_victim)

    return pd.concat(rows)



