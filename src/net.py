from common import nn


class KOCRNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Affine(784, 100))
        self.layers.append(nn.Relu())
        self.layers.append(nn.Affine(100, 10))
        self.layers.append(nn.SoftmaxWithLoss())

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params.append(layer.params)
            self.grads.append(layer.grads)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def gradient(self):
        grads = []
        for layer in self.layers:
            grads.append(layer.grads)
        return grads
