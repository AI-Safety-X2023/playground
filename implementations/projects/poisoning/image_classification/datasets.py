from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms.v2 as T
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
    def from_mnist(cls, dataset: MNIST, device=BEST_DEVICE) -> "EagerDataset":
        data = dataset.data.detach() / 255.0
        transform = T.Normalize((data.mean(dim=(0, 1, 2)),), (data.std(dim=(0, 1, 2)),))
        tensors = [transform(img[None, :]) for img in data]
        
        data = torch.stack(tensors)
        return cls(data, dataset.targets, dataset.classes, device)

    @classmethod
    def from_cifar10(cls, dataset: CIFAR10, device=BEST_DEVICE) -> "EagerDataset":
        targets = torch.tensor(dataset.targets)
        
        transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) #((data.mean(dim=(0, 1, 2)),), (data.std(dim=(0, 1, 2)),)),
        ])
        tensors = [transform(img) for (img, _) in dataset]
        data = torch.stack(tensors)

        return cls(data, targets, dataset.classes, device)

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


def mnist_train_test(root='data') -> tuple[EagerDataset, EagerDataset]:
    """
    Returns the training set and the test set of the MNIST dataset.
    """
    training_data = EagerDataset.from_mnist(
        MNIST(
            root=root,
            train=True,
            download=True,
        ),
    )
    test_data = EagerDataset.from_mnist(
        MNIST(
            root=root,
            train=False,
            download=True,
        ),
    )
    return training_data, test_data

def cifar10_train_test(root='data') -> tuple[EagerDataset, EagerDataset]:
    """
    Returns the training set and the test set of the CIFAR10 dataset.
    """
    training_data = EagerDataset.from_cifar10(
        CIFAR10(
            root=root,
            train=True,
            download=True,
        ),
    )
    test_data = EagerDataset.from_cifar10(
        CIFAR10(
            root=root,
            train=False,
            download=True,
        ),
    )
    return training_data, test_data

def class_weights(data: Dataset):
    """
    Returns the sorted classes and their class weights in a dataset.
    """
    labels = np.array([y.item() for (_, y) in data])
    return np.unique(labels, return_counts=True)