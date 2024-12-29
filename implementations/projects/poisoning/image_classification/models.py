from __future__ import annotations

import torch
from torch import nn

class ConvNet(nn.Module):
    """
    A convolutional neural network classifier.
    """
    def __init__(
            self,
            image_size=28*28,
            num_classes=10,
            num_image_channels=1,
            num_conv_layers=2,
            hidden_layer_dim=256,
            batchnorm=True,
            dropout_rate=0.1,
        ):
        """
        Initialize the network internal state.

        # Parameters

        - `image_size` : the number of pixels in the image
        - `num_classes` : the number of classes for the classifier
        - `num_image_channels` : 1 for a grayscale image, 3 for a RGB image
        - `num_conv_layers` : the number of convolution layers, usually 1 or 2
        - `hidden_layer_dim` : the dimension of the hidden layer in the classifier
        - `batchnorm` : if `True`, add batch normalization after max pooling layers
        - `dropout_rate` : if not `None`, add two dropout layers in the classifier with this rate
        
        The default parameters are set for MNIST digits classification.
        """
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes

        # Each `MaxPool2d(2, 2)` layer divides the number of features by 4
        out_channels = 4 ** num_conv_layers

        # A convolutional neural network that extracts features from the input.
        self.feature_extractor = nn.Sequential()

        for i in range(num_conv_layers):
            in_channels = num_image_channels if i == 0 else out_channels
            self.feature_extractor.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            if batchnorm:
                self.feature_extractor.append(nn.BatchNorm2d(out_channels))
            self.feature_extractor.extend([
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
        
        # A 2-layer MLP
        self.classifier = nn.Sequential(
            nn.Linear(image_size, hidden_layer_dim),
            nn.ReLU(),
        )
        if dropout_rate is not None:
            self.classifier.append(nn.Dropout(dropout_rate))
        self.classifier.append(nn.Linear(hidden_layer_dim, num_classes))
        

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # Convert back logits to CPU for easier data manipulation
        return x.to('cpu')
    
class BigCNN(ConvNet):
    """
    A CNN with two convolution layers.
    """
    def __init__(self, hidden_layer_dim=256, batchnorm=False, dropout_rate=None, **kwargs):
        super().__init__(
            num_conv_layers=2,
            hidden_layer_dim=hidden_layer_dim,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            **kwargs
        )

class SmallCNN(ConvNet):
    """
    A CNN with a single convolution layer.
    """
    def __init__(self, hidden_layer_dim=64, batchnorm=True, dropout_rate=0.1, **kwargs):
        super().__init__(
            num_conv_layers = 1,
            hidden_layer_dim=hidden_layer_dim,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            **kwargs
        )
