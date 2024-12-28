from __future__ import annotations
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset

def plot_pca(dataset: Dataset, n_points=700, classes: Sequence[int] = None):
    """
    If you are running this code on a Jupyter notebook and you want to visualize the PCA
    inside an interactive GUI, you can wrap the function with the following:
    ```python
    %matplotlib tk
    plot_pca(dataset)
    %matplotlib inline
    ```
    """
    # Sample `n_points` from the dataset
    indices = np.random.randint(0, len(dataset), size=n_points)
    features, targets = dataset[indices]

    pca = PCA(n_components=3)

    if classes is not None:
        # Restrict points to the specified classes
        indices = torch.isin(targets, Tensor(classes))
        features = features[indices]
        targets = targets[indices]

    projected = pca.fit_transform(features.cpu().flatten(1))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=30)

    scatter = ax.scatter(
        projected[:, 0],
        projected[:, 1],
        projected[:, 2],
        s=30, c=targets, cmap='tab10',
    )
    plt.legend(*scatter.legend_elements(), title='Classes')
    plt.show()


def display_input_image(input: Tensor, cmap=plt.cm.gray):
    """
    Displays an an input image to a neural network.

    `input`: a 4D tensor
    `cmap`: grayscale by default.
    """
    image = input[0].to('cpu').detach().flatten(0, 1)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap, interpolation='nearest')
    fig.tight_layout()

@torch._dynamo.disable
def _maxpool2d_features_hook(_module, _input, output):
    features = torch.flatten(output, 0, 1).to('cpu').detach()
    ncols = (len(features) + 3) // 4

    fig = plt.figure()
    for i, feature in enumerate(features):
        ax = plt.subplot(4, ncols, i + 1)
        ax.imshow(feature, cmap=plt.cm.gray, interpolation='nearest')
    fig.tight_layout()


def display_cnn_features(model: nn.Module, x: Tensor):
    """
    Displays the intermediate representations of an image
    in a convolutional neural network.

    The features are displayed after each pass through a `MaxPool2d` layer.
    """
    handles = []
    for module in model.modules():
        if isinstance(module, nn.MaxPool2d):
            handle = module.register_forward_hook(_maxpool2d_features_hook)
            handles.append(handle)

    display_input_image(x)

    # When feeding the image to the model, a forward pass is made on the model layers
    # so the hooks will be executed to display internal representations of the input
    model.eval()
    with torch.no_grad():
        model(x)
    
    plt.tight_layout()

    # Clean up
    for handle in handles:
        handle.remove()