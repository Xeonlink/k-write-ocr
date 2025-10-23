# coding: utf-8
from typing import Protocol

import numpy as np

from common import F, util
from constants import PRECISION


class Module(Protocol):
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Relu(Module):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid(Module):
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = F.sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine(Module):
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx


# SoftmaxWithLoss
class SoftmaxWithLoss(Module):
    def __init__(self):
        self.loss = None  # 손실함수
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블(원-핫 인코딩 형태)

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        x = x.reshape(-1, x.shape[-1])
        t = t.reshape(-1, t.shape[-1])

        self.t = t
        self.y = F.softmax(x)
        self.loss = F.cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout: float = 1) -> np.ndarray:
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class Dropout(Module):
    """
    http://arxiv.org/abs/1207.0580
    """

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, is_train=True):
        if is_train:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization(Module):
    """
    http://arxiv.org/abs/1502.03167
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # 합성곱 계층은 4차원, 완전연결 계층은 2차원

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var

        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class Convolution(Module):
    def __init__(self, W: np.ndarray, b: np.ndarray, stride: int = 1, pad: int = 0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 중간 데이터（backward 시 사용）
        self.x = None
        self.col = None
        self.col_W = None

        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = util.im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = util.col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class MaxPooling(Module):
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = util.im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = util.col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


class BatchNorm2d(Module):
    """
    CNN용 배치 정규화 계층
    각 채널별로 독립적으로 정규화를 수행
    """

    def __init__(self, gamma: np.ndarray, beta: np.ndarray, momentum: float = 0.9, running_mean=None, running_var=None):
        """
        Parameters:
            gamma (np.ndarray): 스케일링 파라미터 (C,) 형태
            beta (np.ndarray): 시프팅 파라미터 (C,) 형태
            momentum (float): 이동 평균을 위한 모멘텀 값
            running_mean (np.ndarray): 테스트 시 사용할 평균값
            running_var (np.ndarray): 테스트 시 사용할 분산값
        """
        self.gamma = gamma
        self.beta = beta
        self.num_channels = gamma.shape[0]
        self.momentum = momentum

        # 테스트 시 사용할 통계값
        self.running_mean = running_mean if running_mean is not None else np.zeros(self.num_channels).astype(PRECISION)
        self.running_var = running_var if running_var is not None else np.ones(self.num_channels).astype(PRECISION)

        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x: np.ndarray, train_flg: bool = True) -> np.ndarray:
        """
        Parameters:
            x (np.ndarray): 입력 데이터 (B, C, H, W) 형태
            train_flg (bool): 학습 모드 여부

        Returns:
            np.ndarray: 정규화된 출력 (B, C, H, W) 형태
        """
        if x.ndim != 4:
            raise ValueError("BatchNorm2d는 4차원 입력만 지원합니다")

        B, C, H, W = x.shape

        if C != self.num_channels:
            raise ValueError(f"입력 채널 수({C})가 예상 채널 수({self.num_channels})와 다릅니다")

        # 채널별로 정규화를 위해 (B, C, H*W) 형태로 변환
        x_reshaped = x.reshape(B, C, H * W)

        if train_flg:
            # 각 채널별로 평균과 분산 계산
            mu = x_reshaped.mean(axis=(0, 2))  # (C,) 형태
            xc = x_reshaped - mu.reshape(1, C, 1)  # 브로드캐스팅
            var = np.mean(xc**2, axis=(0, 2))  # (C,) 형태
            std = np.sqrt(var + 1e-7)
            xn = xc / std.reshape(1, C, 1)  # 브로드캐스팅

            # 이동 평균 업데이트
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            # backward를 위한 중간 데이터 저장
            self.batch_size = B
            self.xc = xc
            self.xn = xn
            self.std = std
        else:
            # 테스트 시에는 저장된 통계값 사용
            xc = x_reshaped - self.running_mean.reshape(1, C, 1)
            xn = xc / np.sqrt(self.running_var + 1e-7).reshape(1, C, 1)

        # 스케일링과 시프팅
        out = self.gamma.reshape(1, C, 1) * xn + self.beta.reshape(1, C, 1)

        # 원래 형태로 복원
        return out.reshape(B, C, H, W)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parameters:
            dout (np.ndarray): 출력에 대한 그래디언트 (B, C, H, W) 형태

        Returns:
            np.ndarray: 입력에 대한 그래디언트 (B, C, H, W) 형태
        """
        B, C, H, W = dout.shape

        # 그래디언트를 (B, C, H*W) 형태로 변환
        dout_reshaped = dout.reshape(B, C, H * W)

        # gamma와 beta에 대한 그래디언트 계산
        self.dgamma = np.sum(dout_reshaped * self.xn, axis=(0, 2))
        self.dbeta = np.sum(dout_reshaped, axis=(0, 2))

        # 입력에 대한 그래디언트 계산
        dxn = dout_reshaped * self.gamma.reshape(1, C, 1)

        # xn에 대한 그래디언트를 xc로 변환
        dxc = dxn / self.std.reshape(1, C, 1)

        # xc에 대한 그래디언트를 x로 변환
        dmu = -np.sum(dxc, axis=(0, 2))
        dvar = -np.sum(dxc * self.xc, axis=(0, 2)) / (2 * self.std)

        dx = dxc + dmu.reshape(1, C, 1) / (B * H * W) + dvar.reshape(1, C, 1) * 2 * self.xc / (B * H * W)

        return dx.reshape(B, C, H, W)
