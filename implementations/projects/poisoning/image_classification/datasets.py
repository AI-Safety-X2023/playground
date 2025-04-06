from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset, TensorDataset

from .accel import BEST_DEVICE

class UpdatableDataset(Dataset):
    """A list-like dataset."""
    X: list[Tensor]
    y: list[Tensor]

    def __init__(self):
        self.X = []
        self.y = []

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)
    
    def append(self, X: Tensor, y: Tensor):
        self.X.append(X)
        self.y.append(y)
    
    def extend(self, dataset: Dataset):
        for X, y in dataset:
            self.append(X, y)
    
    def to_tensor_dataset(self) -> TensorDataset:
        return TensorDataset(torch.stack(self.X), torch.stack(self.y))

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
CIFAR10_STD = torch.tensor([0.247, 0.243, 0.261])
CIFAR100_MEAN = torch.tensor([0.5071, 0.4866, 0.4409])
CIFAR100_STD = torch.tensor([0.2673, 0.2564, 0.2762])

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
        data = dataset.data / 255.0
        transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((*data.mean(axis=(0, 1, 2)),), (*data.std(axis=(0, 1, 2)),)),
        ])
        tensors = [transform(img) for (img, _) in dataset]
        data = torch.stack(tensors)

        return cls(data, targets, dataset.classes, device)

    @classmethod
    def from_cifar100(cls, dataset: CIFAR100, device=BEST_DEVICE) -> "EagerDataset":
        return cls.from_cifar10(dataset, device=device)

    def _decode_image(
            self, image: Tensor,
            data_mean: tuple[int], data_std: list[int],
        ) -> np.ndarray:
        image = image.numpy(force=True)
        # Unnormalize image to [0, 1]
        mean = np.array(data_mean).reshape(3, 1, 1)
        std = np.array(data_std).reshape(3, 1, 1)
        image = (std * image) + mean
        # Shuffle dimensions to channels-last format
        return np.transpose(image, (1, 2, 0))
    
    def data_stats(self) -> tuple[Tensor, Tensor]:
        # FIXME: this is brittle
        device = self.data.device
        if len(self.classes) == 100 and self.data.shape[1] == 3:
            return CIFAR10_MEAN.to(device), CIFAR10_STD.to(device)
        elif len(self.classes) == 10 and self.data.shape[1] == 3:
            return CIFAR100_MEAN.to(device), CIFAR100_STD.to(device)
        else:
            raise NotImplementedError("Data stats only implemented for CIFAR")

    def data_range(self) -> tuple[Tensor, Tensor]:
        """
        Returns the min and max RGB values of the normalized dataset.
        """
        mean, std = self.data_stats()
        min = (1. / 255. - mean) / std
        max = (254. / 255. - mean) / std
        return min, max
    
    def clip_to_data_range(self, X: Tensor, inplace=True) -> Tensor:
        """Clip to valid image data range in place."""
        min_, max_ = self.data_range()
        device = X.device
        min_ = min_.reshape(self.data.shape[1], 1, 1).to(device)
        max_ = max_.reshape(self.data.shape[1], 1, 1).to(device)
        if inplace:
            return X.clamp_(min_, max_)
        else:
            return X.clamp(min_, max_)

    def max_data_variation(self) -> Tensor:
        min_, max_ = self.data_range()
        return (max_ - min_).max()

    def decode_cifar10_image(self, image: Tensor) -> np.ndarray:
        """
        Decode a CIFAR-10 image to a `matplotlib`-compatible format.
        """
        return self._decode_image(image, CIFAR10_MEAN, CIFAR10_STD)

    def decode_cifar100_image(self, image: Tensor) -> np.ndarray:
        """
        Decode a CIFAR-100 image to a `matplotlib`-compatible format.
        """
        return self._decode_image(image, CIFAR100_MEAN, CIFAR10_STD)
    
    def random_image_noise(self) -> Tensor:
        rand = torch.rand_like(self.data[0])
        mean, std = self.data_stats()
        mean = mean.reshape(rand.shape[0], 1, 1)
        std = std.reshape(rand.shape[0], 1, 1)
        return mean + std * rand

    def random_label(self) -> Tensor:
        return torch.randint_like(self.targets[0], len(self.classes))
    
    def random_sample_noise(self) -> tuple[Tensor, Tensor]:
        """Generate a random sample `(X, y)` with uniformly sampled coordinates."""
        return self.random_image_noise(), self.random_label()
    
    def decode_target(self, label: Tensor | int) -> str:
        """
        Return the target label's class name in MNIST, CIFAR-10 or CIFAR-100.
        """
        if isinstance(label, Tensor):
            label = label.item()
        return self.classes[label]
    

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
    Returns the training set and the test set of the CIFAR-10 dataset.
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

def cifar100_train_test(root='data') -> tuple[EagerDataset, EagerDataset]:
    """
    Returns the training set and the test set of the CIFAR-100 dataset.
    """
    training_data = EagerDataset.from_cifar100(
        CIFAR100(
            root=root,
            train=True,
            download=True,
        ),
    )
    test_data = EagerDataset.from_cifar100(
        CIFAR100(
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