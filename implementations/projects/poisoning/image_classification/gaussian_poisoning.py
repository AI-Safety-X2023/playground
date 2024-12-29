from __future__ import annotations

from copy import deepcopy
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, TensorDataset, Subset, DataLoader

class GaussianPoisoningDataset(TensorDataset):
    """
    A custom dataset that adds gaussian centered noise to the features of a dataset.

    You can access the original, untouched dataset with the field `self.clean_dataset`.
    """

    def __init__(
            self,
            dataset: Dataset,
            poison_budget: float,
            noise_std: float,
            device=None,
        ):
        """
        Create a new `GaussianPoisoningDataset`.

        # Parameters

        - `dataset` : the original dataset, which is assumed to have a field `targets`
        - `poison_budget` : the proportion of the data to be poisoned
        - `noise_std` : the standard deviation of the noise.
        """
        if device is None:
            device = dataset.data.device

        self.clean_dataset = dataset
        size = len(dataset)

        # A boolean array that indicates the samples to poison
        self.poison_support = (torch.rand(size, device=device) < poison_budget).bool()

        # Generate gaussian noise over the poison support, and zero outside
        self.noise = torch.zeros_like(dataset.data, device=device)
        gaussians = torch.normal(
            mean=0.0,
            std=noise_std,
            size=(self.num_poisons(), *dataset.data[0].shape),
            device=device,
        )
        self.noise[self.poison_support] = gaussians
        
        self.poison_budget = poison_budget
        self.noise_std = noise_std

        # The corrupted data
        self.data = dataset.data.to(device) + self.noise
        # Targets are untouched and aliased
        self.targets = dataset.targets

        super().__init__(self.data, self.targets)

    def num_poisons(self) -> int:
        """
        The number of poisoned samples in this dataset.

        WARNING: this is not exactly equal to `int(poison_budget * len(clean_dataset))`.
        """
        return self.poison_support.int().sum().item()

    def _poison_indices(self) -> Tensor:
        return torch.nonzero(self.poison_support, as_tuple=True)[0]

    def _clean_indices(self) -> Tensor:
        return torch.nonzero(self.poison_support.logical_not(), as_tuple=True)[0]

    def poisoned_subset(self) -> Subset:
        """
        Returns the poisoned subset of this dataset.

        This subset contains items of the form `(X_poison, y)`.
        """
        return Subset(self, self._poison_indices())
    
    def clean_subset(self) -> Subset:
        """
        Returns the clean subset of this dataset.

        This subset contains items of the form `(X_clean, y)`.
        """
        return Subset(self, self._clean_indices())
    
    def clean_subset_before_poisoning(self) -> Subset:
        """
        Returns the clean version of the poisoned subset of this dataset.

        This subset contains items of the form `(X_base, y)`, where
        `X_base = X_poison - noise`.
        """
        return Subset(self.clean_dataset, self._poison_indices())

    def noise_of_poisoned_subset(self) -> Tensor:
        """
        Returns the noise used to corrupt the poisoned subset of this dataset.

        The noise is returned as a tensor with the same length as the number
        of poisoned samples.
        """
        return self.noise[self.poison_support]

def gaussian_unlearning_score(
        model: nn.Module,
        base_data: Dataset,
        noise: Dataset,
        noise_std: float,
        loss_fn: _Loss,
        batch_size=1,
        epsilon=1e-8,
    ) -> Tensor:
    """
    Compute the gaussian unlearning score (GUS) for each element of a dataset.

    ## Parameters
    
    - `base_data` : the **clean** data points `(x_base, y)` such that
        the model was trained on `(x, y)` where `x = x_base + noise`.
    - `noise` : the gaussian centered noise values to test on.
        Usually the noise that was used to poison `base_data`.
    - `noise_std` : the standard deviation of the noise distribution.
    - `loss_fn` : the loss function used to compute the gradients.
        This is the criterion used to train the `model`.
    - `batch_size` : a performance setting, not relevant at the moment.
    - `epsilon` : a small value to avoid divisions by zero.
      
    ## Notes

    The aggregated GUS is the mean of the individual scores.

    We follow the _Algorithm 3_ in the original paper of Pawelczyk et al:
    https://arxiv.org/abs/2406.17216
    """
    p = len(base_data)
    assert p == len(noise), f"Dataset lengths don't match: {p}, {len(noise)}"

    base_loader = DataLoader(base_data, batch_size=batch_size)
    noise_loader = DataLoader(noise, batch_size=batch_size)

    I_poison = torch.zeros(len(base_data))
    i = 0
    model.eval()

    # There is no memory leak since these dataloader have same length
    for (X_b, y_b), noise_b in zip(base_loader, noise_loader):
        for X_base, y, xi in zip(X_b, y_b, noise_b):

            # TODO: compute the model predictions in batch for performance
            # instead of element-by-element computation, and increase batch_size
            # (this requires to be careful with gradient computation)

            X_base.requires_grad_(True)
            loss = loss_fn(model(X_base), y)
            loss.backward()

            g = X_base.grad
            g_n = g.norm()

            I_poison[i] = xi.dot(g) / (epsilon + noise_std * g_n)
            X_base.requires_grad_(False)
            i += 1
    return I_poison