from collections import OrderedDict

import numpy as np

from common import nn


class KOCRNet(nn.Module):

    def __init__(
        self,
        input_shape: tuple = (100, 1, 260, 660),
        output_shape: tuple = (100, 10, 3, 28),
        weight_init_std: float = 0.01,
    ):
        super().__init__()

        B, C, H, W = input_shape
        B, L, M, S = self.output_shape = output_shape

        self.params = {}
        self.layers = OrderedDict[str, nn.Module]()
        # shape: (B, C, H, W)

        # Conv0
        conv0_channels = 32
        self.params["Conv0.W"] = weight_init_std * np.random.randn(conv0_channels, C, 3, 3)
        self.params["Conv0.b"] = np.zeros(conv0_channels)
        self.layers["Conv0"] = nn.Convolution(self.params["Conv0.W"], self.params["Conv0.b"], stride=1, pad=1)
        # shape: (B, conv0_channels, H, W)

        # Relu0
        self.layers["Relu0"] = nn.Relu()
        # shape: (B, conv0_channels, H, W)

        # Pool0
        self.layers["Pool0"] = nn.MaxPooling(2, 2, stride=2)
        # shape: (B, conv0_channels, H/2, W/2)

        # Conv1
        conv1_channels = 32
        self.params["Conv1.W"] = weight_init_std * np.random.randn(conv1_channels, conv0_channels, 3, 3)
        self.params["Conv1.b"] = np.zeros(conv1_channels)
        self.layers["Conv1"] = nn.Convolution(self.params["Conv1.W"], self.params["Conv1.b"], stride=1, pad=1)
        # shape: (B, conv1_channels, H/2, W/2)

        # Relu1
        self.layers["Relu1"] = nn.Relu()
        # shape: (B, conv1_channels, H/2, W/2)

        # Pool1
        self.layers["Pool1"] = nn.MaxPooling(2, 2, stride=2)
        # shape: (B, conv1_channels, H/4, W/4)

        # Conv2
        conv2_channels = 32
        self.params["Conv2.W"] = weight_init_std * np.random.randn(conv2_channels, conv1_channels, 3, 3)
        self.params["Conv2.b"] = np.zeros(conv2_channels)
        self.layers["Conv2"] = nn.Convolution(self.params["Conv2.W"], self.params["Conv2.b"], stride=1, pad=1)
        # shape: (B, conv2_channels, H/4, W/4)

        # Relu2
        self.layers["Relu2"] = nn.Relu()
        # shape: (B, conv2_channels, H/4, W/4)

        # Pool2
        self.layers["Pool2"] = nn.MaxPooling(2, 2, stride=2)
        # shape: (B, conv2_channels, H/8, W/8)

        # Conv3
        conv3_channels = 32
        self.params["Conv3.W"] = weight_init_std * np.random.randn(conv3_channels, conv2_channels, 3, 3)
        self.params["Conv3.b"] = np.zeros(conv3_channels)
        self.layers["Conv3"] = nn.Convolution(self.params["Conv3.W"], self.params["Conv3.b"], stride=1, pad=1)
        # shape: (B, conv3_channels, H/8, W/8)

        # Relu3
        self.layers["Relu3"] = nn.Relu()
        # shape: (B, conv3_channels, H/8, W/8)

        # Pool3
        self.layers["Pool3"] = nn.MaxPooling(2, 2, stride=2)
        # shape: (B, conv3_channels, H/16, W/16)

        # Conv4
        conv4_channels = 32
        self.params["Conv4.W"] = weight_init_std * np.random.randn(conv4_channels, conv3_channels, 3, 3)
        self.params["Conv4.b"] = np.zeros(conv4_channels)
        self.layers["Conv4"] = nn.Convolution(self.params["Conv4.W"], self.params["Conv4.b"], stride=1, pad=1)
        # shape: (B, conv4_channels, H/16, W/16)

        # Relu4
        self.layers["Relu4"] = nn.Relu()
        # shape: (B, conv4_channels, H/16, W/16)

        # Pool4
        self.layers["Pool4"] = nn.MaxPooling(2, 2, stride=2)
        # shape: (B, conv4_channels, H/32, W/32)

        # Conv5
        conv5_channels = 32
        self.params["Conv5.W"] = weight_init_std * np.random.randn(conv5_channels, conv4_channels, 3, 3)
        self.params["Conv5.b"] = np.zeros(conv5_channels)
        self.layers["Conv5"] = nn.Convolution(self.params["Conv5.W"], self.params["Conv5.b"], stride=1, pad=1)
        # shape: (B, conv5_channels, H/32, W/32)

        # Relu5
        self.layers["Relu5"] = nn.Relu()
        # shape: (B, conv5_channels, H/32, W/32)

        # Pool5
        self.layers["Pool5"] = nn.MaxPooling(2, 2, stride=2)
        # shape: (B, conv5_channels, H/64, W/64)

        # Affine0
        affine0_input_size = conv5_channels * int(H / 64) * int(W / 64)
        affine0_hidden = 1000
        self.params["Affine0.W"] = weight_init_std * np.random.randn(affine0_input_size, affine0_hidden)
        self.params["Affine0.b"] = np.zeros(affine0_hidden)
        self.layers["Affine0"] = nn.Affine(self.params["Affine0.W"], self.params["Affine0.b"])
        # shape: (B, affine0_hidden)

        # Dropout0
        self.layers["Dropout0"] = nn.Dropout(dropout_ratio=0.5)
        # shape: (B, affine0_hidden)

        # Affine1
        affine1_input_size = affine0_hidden
        affine1_hidden = L * M * S
        self.params["Affine1.W"] = weight_init_std * np.random.randn(affine1_input_size, affine1_hidden)
        self.params["Affine1.b"] = np.zeros(affine1_hidden)
        self.layers["Affine1"] = nn.Affine(self.params["Affine1.W"], self.params["Affine1.b"])
        # shape: (B, affine1_hidden)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers.values():
            x = layer.forward(x)

        B, L, M, S = self.output_shape
        x = x.reshape(B, L, M, S)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        B, L, M, S = self.output_shape
        dout = dout.reshape(B, L * M * S)

        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

    def gradient(self) -> dict[str, np.ndarray]:
        grads = {}
        grads["Conv0.W"] = self.layers["Conv0"].dW
        grads["Conv0.b"] = self.layers["Conv0"].db
        grads["Conv1.W"] = self.layers["Conv1"].dW
        grads["Conv1.b"] = self.layers["Conv1"].db
        grads["Conv2.W"] = self.layers["Conv2"].dW
        grads["Conv2.b"] = self.layers["Conv2"].db
        grads["Conv3.W"] = self.layers["Conv3"].dW
        grads["Conv3.b"] = self.layers["Conv3"].db
        grads["Conv4.W"] = self.layers["Conv4"].dW
        grads["Conv4.b"] = self.layers["Conv4"].db
        grads["Conv5.W"] = self.layers["Conv5"].dW
        grads["Conv5.b"] = self.layers["Conv5"].db
        grads["Affine0.W"] = self.layers["Affine0"].dW
        grads["Affine0.b"] = self.layers["Affine0"].db
        grads["Affine1.W"] = self.layers["Affine1"].dW
        grads["Affine1.b"] = self.layers["Affine1"].db
        return grads
