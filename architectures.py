"""
Architectures file of example project.
"""

import torch
import torchvision
from segmentation.models import all_models


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimpleCNN(torch.nn.Module):
    def __init__(self, n_in_channels: int = 2, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super(SimpleCNN, self).__init__()

        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(in_channels=n_in_channels, out_channels=n_kernels, kernel_size=kernel_size,
                                       bias=True, padding=int(kernel_size / 2)))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)
        self.output_layer = torch.nn.Conv2d(in_channels=n_in_channels, out_channels=1,
                                            kernel_size=kernel_size, bias=True, padding=int(kernel_size / 2))

    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        return pred


class DenseCNN(torch.nn.Module):
    def __init__(self, n_input_channels: int = 2, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        """CNN, consisting of `n_hidden_layers` linear layers, using relu
        activation function in the hidden CNN layers.

        Parameters
        ----------
        n_input_channels: int
            Number of features channels in input tensor
        n_hidden_layers: int
            Number of conv. layers
        n_kernels: int
            Number of kernels in each layer
        kernel_size: int
            Number of features in output tensor
        """
        super(DenseCNN, self).__init__()

        super().__init__()
        layers = []
        n_concat_channels = n_input_channels
        for i in range(n_hidden_layers):
            # Add a CNN layer
            layer = torch.nn.Conv2d(in_channels=n_concat_channels,
                                    out_channels=n_kernels,
                                    kernel_size=kernel_size,
                                    padding=int(kernel_size / 2))
            layers.append(layer)
            self.add_module(f"conv_{i:03d}", layer)
            # Prepare for concatenated input
            n_concat_channels = n_kernels + n_input_channels
            n_input_channels = n_kernels

        self.layers = layers

    def forward(self, x):
        """Apply CNN to `x`

        Parameters
        ----------
        x: torch.tensor
            Input tensor of shape (n_samples, n_input_channels, x, y)

        Returns
        ----------
        torch.tensor
            Output tensor of shape (n_samples, n_output_channels, u, v)
        """
        # Apply layers module
        skip_connection = None
        output = None
        for layer in self.layers:
            # If previous output and skip_connection exist, concatenate
            # them and store previous output as new skip_connection. Otherwise,
            # use x as input and store it as skip_connection.
            if skip_connection is not None:
                inp = torch.cat([output, skip_connection], dim=1)
                skip_connection = output
            else:
                inp = x
                skip_connection = x
            # Apply CNN layer
            output = torch.relu_(layer(inp))

        return output


class ResCNN(torch.nn.Module):
    def __init__(self, n_input_channels: int = 2, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        """CNN, consisting of `n_hidden_layers` linear layers, using relu
        activation function in the hidden CNN layers.

        Parameters
        ----------
        n_input_channels: int
            Number of features channels in input tensor
        n_hidden_layers: int
            Number of conv. layers
        n_kernels: int
            Number of kernels in each layer
        kernel_size: int
            Number of features in output tensor
        """
        super(ResCNN, self).__init__()

        super().__init__()
        layers = []
        n_concat_channels = n_input_channels
        for i in range(n_hidden_layers):
            # Add a CNN layer
            layer = torch.nn.Conv2d(in_channels=n_concat_channels,
                                    out_channels=n_kernels,
                                    kernel_size=kernel_size,
                                    padding=int(kernel_size / 2))
            layers.append(layer)
            self.add_module(f"conv_{i:03d}", layer)
            # Prepare for concatenated input
            n_concat_channels = n_kernels + n_input_channels
            n_input_channels = n_kernels

        self.layers = layers

    def forward(self, x):
        """Apply CNN to `x`

        Parameters
        ----------
        x: torch.tensor
            Input tensor of shape (n_samples, n_input_channels, x, y)

        Returns
        ----------
        torch.tensor
            Output tensor of shape (n_samples, n_output_channels, u, v)
        """
        # Apply layers module
        skip_connection = None
        output = None
        for layer in self.layers:
            # If previous output and skip_connection exist, concatenate
            # them and store previous output as new skip_connection. Otherwise,
            # use x as input and store it as skip_connection.
            if skip_connection is not None:
                inp = output + skip_connection
                skip_connection = output
            else:
                inp = x
                skip_connection = x
            # Apply CNN layer
            output = torch.relu_(layer(inp))

        return output


class FCN_ResNet50(torch.nn.Module):
    def __init__(self):
        super(FCN_ResNet50, self).__init__()
        self.model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=1)
        self.model.backbone.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                    bias=False)  # TODO: get rid of this?
        # model.classifier = Identity()
        self.model.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))

    def get(self):
        return self.model


class fcn8_resnet34(torch.nn.Module):
    def __init__(self, batch_size, n_classes: int = 1):
        super(fcn8_resnet34, self).__init__()
        self.model = all_models.model_from_name['fcn32_resnet50'](n_classes, batch_size)
        self.model.features[0] = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                 bias=False)
        self.model.classifier[1] = Identity()

    def get(self):
        return self.model


class pretrained_resnet(torch.nn.Module):
    def __init__(self):
        super(pretrained_resnet, self).__init__()
        self.arch = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
        self.arch.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        self.arch.aux_classifier = Identity()
        ct = 0
        for child in self.arch.children():
            ct += 1
            if ct < 2:
                for param in child.parameters():
                    param.requires_grad = False

    def get(self):
        return self.arch
