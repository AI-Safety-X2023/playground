from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import VisionDataset, MNIST
from torchvision.transforms.functional import to_tensor as image_to_tensor
from torch.utils.data import Dataset, TensorDataset

from .accel import BEST_DEVICE


class EagerDataset(TensorDataset):
    """
    A dataset that eagerly performs preprocessing on a dataset.

    By default, this dataset sends all the features to the GPU if available.
    This eliminates any performance benefit of writing
    `Dataloader(dataset, num_workers=...)` since no preprocessing is done.

    If the dataset is too large for your GPU, you should add the argument
    `device='cpu'` in order to avoid memory errors. In that case, consider
    setting `num_workers` and `pin_memory=True` to `Dataloader`.
    """
    def __init__(
            self,
            data: Tensor,
            targets: Tensor,
            classes: list[str] = None,
            device=BEST_DEVICE,
        ):
        """
        Create a new dataset.
        
        If `classes` is a list of class names, which is relevant
        if you want to make classification data.
        """
        self.classes = classes
        if classes is None:
            self.classes = targets.unique()

        self.data = data.to(device)
        if len(targets.shape) == 1:
            # This makes it easier to when dealing with classifiers
            targets = targets.to('cpu', copy=True)
        self.targets = targets

        super().__init__(self.data, self.targets)
    
    @classmethod
    def from_torchvision(cls, dataset: VisionDataset, device=BEST_DEVICE) -> "EagerDataset":
        tensors = [image_to_tensor(img) for (img, _) in dataset]
        data = torch.stack(tensors)
        return cls(data, dataset.targets, dataset.classes, device)

    def split(
            self,
            split_size_or_sections: int | list[int]
        ) -> list["EagerDataset"]:
        """
        Split this dataset into chunks.

        Returns a list of datasets referencing the same tensors.
        See `torch.split` for more information.
        """
        data_list = self.data.split(split_size_or_sections)
        targets_list = self.targets.split(split_size_or_sections)
        return [
            EagerDataset(data, targets, self.classes)
            for (data, targets) in zip(data_list, targets_list)
        ]


mnist_training_data = EagerDataset.from_torchvision(
    MNIST(
        root='data',
        train=True,
        download=True,
    ),
)

mnist_test_data = EagerDataset.from_torchvision(
    MNIST(
        root='data',
        train=False,
        download=True,
    ),
)


def class_weights(data: Dataset):
    """
    Returns the sorted classes and their class weights in a dataset.
    """
    labels = np.array([y.item() for (_, y) in data])
    return np.unique(labels, return_counts=True)