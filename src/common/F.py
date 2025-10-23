# coding: utf-8
import numpy as np


def identity_function(x: np.ndarray) -> np.ndarray:
    return x


def step_function(x: np.ndarray) -> np.ndarray:
    return np.array(x > 0, dtype=np.int)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


def softmax(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y: np.ndarray, t: np.ndarray) -> float:
    return 0.5 * np.sum((y - t) ** 2)


def mean_squared_error_fixed(y: np.ndarray, t: np.ndarray) -> float:
    return np.mean((y - t) ** 2)


def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X: np.ndarray, t: np.ndarray) -> float:
    y = softmax(X)
    return cross_entropy_error(y, t)
