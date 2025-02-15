from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5,
    squeezenet1_1, mobilenet_v3_small, mobilenet_v3_large,
    resnet18, resnet34, resnet50, resnet101, resnet152
)

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

        - image_size: the number of pixels in the image
        - num_classes: the number of classes for the classifier
        - num_image_channels: 1 for a grayscale image, 3 for a RGB image
        - num_conv_layers: the number of convolution layers, usually 1 or 2
        - hidden_layer_dim: the dimension of the hidden layer in the classifier
        - batchnorm: if `True`, add batch normalization after max pooling layers
        - dropout_rate: if not `None`, add two dropout layers in the classifier with this rate
        
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
            num_conv_layers=1,
            hidden_layer_dim=hidden_layer_dim,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            **kwargs
        )


class ResNet(nn.Module):
    """
    ResNet.
    """
    def __init__(self, num_layers=18, num_classes=10):
        super().__init__()
        models = {
            18: resnet18,
            34: resnet34,
            50: resnet50,
            101: resnet101,
            152: resnet152,
        }
        try:
            model = models[num_layers]
        except KeyError:
            raise ValueError(f"Invalid number of layers: {num_layers}. Expected value in {list(models)}")
        self.resnet = model(num_classes=num_classes)
    
    def forward(self, x):
        return self.resnet.forward(x).to('cpu')

class MobileNetV3(nn.Module):
    """
    MobileNet v3.
    """
    def __init__(self, size: str = 'small', num_classes=10):
        super().__init__()
        models = {
            'small': mobilenet_v3_small,
            'large': mobilenet_v3_large,
        }
        try:
            fn = models[size]
        except KeyError:
            raise ValueError(f"Invalid size: {size}. Expected value in {list(models)}")
        self.mobilenet = fn(num_classes=num_classes)
    
    def forward(self, x):
        return self.mobilenet.forward(x).to('cpu')

class SqueezeNet1_1(nn.Module):
    """
    SqueezeNet v1.1.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.squeezenet = squeezenet1_1(num_classes=num_classes)
    
    def forward(self, x):
        return self.squeezenet.forward(x).to('cpu')

class EfficientNet(nn.Module):
    """
    EfficientNet.
    """
    def __init__(self, b=0, num_classes=10):
        super().__init__()
        models = [
            efficientnet_b0,
            efficientnet_b1,
            efficientnet_b2,
            efficientnet_b3,
            efficientnet_b4,
            efficientnet_b5,
        ]
        try:
            model = models[b]
        except KeyError:
            raise ValueError(f"Invalid EfficientNet size: {b}. Expected value in {list(range(0, 6))}")
        self.resnet = model(num_classes=num_classes)
    
    def forward(self, x):
        return self.resnet.forward(x).to('cpu')
