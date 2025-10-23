from collections import OrderedDict

import numpy as np

from common import nn
from constants import PRECISION


class KOCRNet(nn.Module):

    def __init__(
        self,
        input_shape: tuple = (1, 260, 660),  # shape without batch size
        output_shape: tuple = (10, 3, 28),  # shape without batch size
        weight_init_std: float = 0.01,
    ):
        super().__init__()

        C, H, W = input_shape
        L, M, S = self.output_shape = output_shape

        self.params = {}
        self.layers = OrderedDict[str, nn.Module]()
        # shape: (B, C, H, W)

        # Block 0: Conv - Relu - Pool
        conv0_channels = 32
        std = np.sqrt(2.0 / (1 * 3 * 3))
        self.params["Conv0.W"] = std * np.random.randn(conv0_channels, C, 3, 3).astype(PRECISION)
        self.params["Conv0.b"] = np.zeros(conv0_channels).astype(PRECISION)
        self.layers["Conv0"] = nn.Convolution(self.params["Conv0.W"], self.params["Conv0.b"], stride=1, pad=1)
        self.params["BatchNorm0.gamma"] = np.ones(conv0_channels).astype(PRECISION)
        self.params["BatchNorm0.beta"] = np.zeros(conv0_channels).astype(PRECISION)
        self.layers["BatchNorm0"] = nn.BatchNorm2d(self.params["BatchNorm0.gamma"], self.params["BatchNorm0.beta"])
        self.layers["Relu0"] = nn.Relu()
        self.layers["Pool0"] = nn.MaxPooling(2, 2, stride=2)
        # shape after block 0: (B, conv0_channels, H/2, W/2)

        # Block 1: Conv - Relu - Pool
        conv1_channels = 32
        std = np.sqrt(2.0 / (conv0_channels * 3 * 3))
        self.params["Conv1.W"] = std * np.random.randn(conv1_channels, conv0_channels, 3, 3).astype(PRECISION)
        self.params["Conv1.b"] = np.zeros(conv1_channels).astype(PRECISION)
        self.layers["Conv1"] = nn.Convolution(self.params["Conv1.W"], self.params["Conv1.b"], stride=1, pad=1)
        self.params["BatchNorm1.gamma"] = np.ones(conv1_channels).astype(PRECISION)
        self.params["BatchNorm1.beta"] = np.zeros(conv1_channels).astype(PRECISION)
        self.layers["BatchNorm1"] = nn.BatchNorm2d(self.params["BatchNorm1.gamma"], self.params["BatchNorm1.beta"])
        self.layers["Relu1"] = nn.Relu()
        self.layers["Pool1"] = nn.MaxPooling(2, 2, stride=2)
        # shape after block 1: (B, conv1_channels, H/4, W/4)

        # Block 2: Conv - Relu - Pool
        conv2_channels = 32
        std = np.sqrt(2.0 / (conv1_channels * 3 * 3))
        self.params["Conv2.W"] = std * np.random.randn(conv2_channels, conv1_channels, 3, 3).astype(PRECISION)
        self.params["Conv2.b"] = np.zeros(conv2_channels).astype(PRECISION)
        self.layers["Conv2"] = nn.Convolution(self.params["Conv2.W"], self.params["Conv2.b"], stride=1, pad=1)
        self.params["BatchNorm2.gamma"] = np.ones(conv2_channels).astype(PRECISION)
        self.params["BatchNorm2.beta"] = np.zeros(conv2_channels).astype(PRECISION)
        self.layers["BatchNorm2"] = nn.BatchNorm2d(self.params["BatchNorm2.gamma"], self.params["BatchNorm2.beta"])
        self.layers["Relu2"] = nn.Relu()
        self.layers["Pool2"] = nn.MaxPooling(2, 2, stride=2)
        # shape after block 2: (B, conv2_channels, H/8, W/8)

        # Block 3: Conv - Relu - Pool
        conv3_channels = 32
        std = np.sqrt(2.0 / (conv2_channels * 3 * 3))
        self.params["Conv3.W"] = std * np.random.randn(conv3_channels, conv2_channels, 3, 3).astype(PRECISION)
        self.params["Conv3.b"] = np.zeros(conv3_channels).astype(PRECISION)
        self.layers["Conv3"] = nn.Convolution(self.params["Conv3.W"], self.params["Conv3.b"], stride=1, pad=1)
        self.params["BatchNorm3.gamma"] = np.ones(conv3_channels).astype(PRECISION)
        self.params["BatchNorm3.beta"] = np.zeros(conv3_channels).astype(PRECISION)
        self.layers["BatchNorm3"] = nn.BatchNorm2d(self.params["BatchNorm3.gamma"], self.params["BatchNorm3.beta"])
        self.layers["Relu3"] = nn.Relu()
        self.layers["Pool3"] = nn.MaxPooling(2, 2, stride=2)
        # shape after block 3: (B, conv3_channels, H/16, W/16)

        # Block 4: Conv - Relu - Pool
        conv4_channels = 32
        std = np.sqrt(2.0 / (conv3_channels * 3 * 3))
        self.params["Conv4.W"] = std * np.random.randn(conv4_channels, conv3_channels, 3, 3).astype(PRECISION)
        self.params["Conv4.b"] = np.zeros(conv4_channels).astype(PRECISION)
        self.layers["Conv4"] = nn.Convolution(self.params["Conv4.W"], self.params["Conv4.b"], stride=1, pad=1)
        self.params["BatchNorm4.gamma"] = np.ones(conv4_channels).astype(PRECISION)
        self.params["BatchNorm4.beta"] = np.zeros(conv4_channels).astype(PRECISION)
        self.layers["BatchNorm4"] = nn.BatchNorm2d(self.params["BatchNorm4.gamma"], self.params["BatchNorm4.beta"])
        self.layers["Relu4"] = nn.Relu()
        self.layers["Pool4"] = nn.MaxPooling(2, 2, stride=2)
        # shape after block 4: (B, conv4_channels, H/32, W/32)

        # Block 5: Conv - Relu - Pool
        conv5_channels = 32
        std = np.sqrt(2.0 / (conv4_channels * 3 * 3))
        self.params["Conv5.W"] = std * np.random.randn(conv5_channels, conv4_channels, 3, 3).astype(PRECISION)
        self.params["Conv5.b"] = np.zeros(conv5_channels).astype(PRECISION)
        self.layers["Conv5"] = nn.Convolution(self.params["Conv5.W"], self.params["Conv5.b"], stride=1, pad=1)
        self.params["BatchNorm5.gamma"] = np.ones(conv5_channels).astype(PRECISION)
        self.params["BatchNorm5.beta"] = np.zeros(conv5_channels).astype(PRECISION)
        self.layers["BatchNorm5"] = nn.BatchNorm2d(self.params["BatchNorm5.gamma"], self.params["BatchNorm5.beta"])
        self.layers["Relu5"] = nn.Relu()
        self.layers["Pool5"] = nn.MaxPooling(2, 2, stride=2)
        # shape after block 5: (B, conv5_channels, H/64, W/64)

        # Block 6: Conv - Relu - Pool
        conv6_channels = 32
        std = np.sqrt(2.0 / (conv5_channels * 3 * 3))
        self.params["Conv6.W"] = std * np.random.randn(conv6_channels, conv5_channels, 3, 3).astype(PRECISION)
        self.params["Conv6.b"] = np.zeros(conv6_channels).astype(PRECISION)
        self.layers["Conv6"] = nn.Convolution(self.params["Conv6.W"], self.params["Conv6.b"], stride=1, pad=1)
        self.params["BatchNorm6.gamma"] = np.ones(conv6_channels).astype(PRECISION)
        self.params["BatchNorm6.beta"] = np.zeros(conv6_channels).astype(PRECISION)
        self.layers["BatchNorm6"] = nn.BatchNorm2d(self.params["BatchNorm6.gamma"], self.params["BatchNorm6.beta"])
        self.layers["Relu6"] = nn.Relu()
        self.layers["Pool6"] = nn.MaxPooling(2, 2, stride=2)
        # shape after block 6: (B, conv6_channels, H/128, W/128)

        # Affine0
        affine0_input_size = conv6_channels * int(H / 128) * int(W / 128)
        affine0_hidden = 1000
        std = np.sqrt(2.0 / (conv6_channels * 3 * 3))
        self.params["Affine0.W"] = std * np.random.randn(affine0_input_size, affine0_hidden).astype(PRECISION)
        self.params["Affine0.b"] = np.zeros(affine0_hidden).astype(PRECISION)
        self.layers["Affine0"] = nn.Affine(self.params["Affine0.W"], self.params["Affine0.b"])
        # shape: (B, affine0_hidden)

        # Dropout0
        self.layers["Dropout0"] = nn.Dropout(dropout_ratio=0.8)
        # shape: (B, affine0_hidden)

        # Affine1
        affine1_input_size = affine0_hidden
        affine1_hidden = L * M * S
        std = np.sqrt(2.0 / affine1_hidden)
        self.params["Affine1.W"] = std * np.random.randn(affine1_input_size, affine1_hidden).astype(PRECISION)
        self.params["Affine1.b"] = np.zeros(affine1_hidden).astype(PRECISION)
        self.layers["Affine1"] = nn.Affine(self.params["Affine1.W"], self.params["Affine1.b"])
        # shape: (B, affine1_hidden)

    def forward(self, x: np.ndarray, is_train: bool = True) -> np.ndarray:
        self.batch_size = x.shape[0]  # batch size

        for name, layer in self.layers.items():
            if name.startswith("Dropout"):
                x = layer.forward(x, is_train)
            elif name.startswith("BatchNorm"):
                x = layer.forward(x, is_train)
            else:
                x = layer.forward(x)

        L, M, S = self.output_shape
        x = x.reshape(self.batch_size, L, M, S)
        return x

    def backward(self, dout: np.ndarray) -> None:
        L, M, S = self.output_shape
        dout = dout.reshape(self.batch_size, L * M * S)

        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

    def gradient(self) -> dict[str, np.ndarray]:
        grads = {}
        grads["Conv0.W"] = self.layers["Conv0"].dW
        grads["Conv0.b"] = self.layers["Conv0"].db
        grads["BatchNorm0.gamma"] = self.layers["BatchNorm0"].dgamma
        grads["BatchNorm0.beta"] = self.layers["BatchNorm0"].dbeta
        grads["Conv1.W"] = self.layers["Conv1"].dW
        grads["Conv1.b"] = self.layers["Conv1"].db
        grads["BatchNorm1.gamma"] = self.layers["BatchNorm1"].dgamma
        grads["BatchNorm1.beta"] = self.layers["BatchNorm1"].dbeta
        grads["Conv2.W"] = self.layers["Conv2"].dW
        grads["Conv2.b"] = self.layers["Conv2"].db
        grads["BatchNorm2.gamma"] = self.layers["BatchNorm2"].dgamma
        grads["BatchNorm2.beta"] = self.layers["BatchNorm2"].dbeta
        grads["Conv3.W"] = self.layers["Conv3"].dW
        grads["Conv3.b"] = self.layers["Conv3"].db
        grads["BatchNorm3.gamma"] = self.layers["BatchNorm3"].dgamma
        grads["BatchNorm3.beta"] = self.layers["BatchNorm3"].dbeta
        grads["Conv4.W"] = self.layers["Conv4"].dW
        grads["Conv4.b"] = self.layers["Conv4"].db
        grads["BatchNorm4.gamma"] = self.layers["BatchNorm4"].dgamma
        grads["BatchNorm4.beta"] = self.layers["BatchNorm4"].dbeta
        grads["Conv5.W"] = self.layers["Conv5"].dW
        grads["Conv5.b"] = self.layers["Conv5"].db
        grads["BatchNorm5.gamma"] = self.layers["BatchNorm5"].dgamma
        grads["BatchNorm5.beta"] = self.layers["BatchNorm5"].dbeta
        grads["Conv6.W"] = self.layers["Conv6"].dW
        grads["Conv6.b"] = self.layers["Conv6"].db
        grads["BatchNorm6.gamma"] = self.layers["BatchNorm6"].dgamma
        grads["BatchNorm6.beta"] = self.layers["BatchNorm6"].dbeta
        grads["Affine0.W"] = self.layers["Affine0"].dW
        grads["Affine0.b"] = self.layers["Affine0"].db
        grads["Affine1.W"] = self.layers["Affine1"].dW
        grads["Affine1.b"] = self.layers["Affine1"].db

        return grads
