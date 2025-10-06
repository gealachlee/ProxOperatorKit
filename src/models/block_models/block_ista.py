# ** coding: utf-8 **
"""
@Author: Zhihong Li, Rongrong Lin
@date: 2024/11/25
@description: ISTA.
@version: 2.0
"""
from typing import Callable

import numpy as np

from models.base import Model


class ISTA(Model):

    def __init__(self, A: np.ndarray, tau, prox_func: Callable):
        super().__init__(A, tau)
        self.m, self.n = self.A.shape
        self.L_np = np.linalg.norm(np.matmul(A.transpose(), A), ord=2)
        self.gamma = 1 / np.linalg.norm(A, 2) ** 2
        self.prox_func: Callable = prox_func
        self.iter_history: list = []

    def name(self):
        return f'ISTA with {self.prox_func.__name__}'

    def T(self, x, d, **kwargs):
        tau = kwargs.get('tau', self.tau)
        index = kwargs.get('index', -1)
        assert index >= 0

        r = (self.A @ x.T).T - d
        z = x - self.gamma*2 * (self.A.T @ r.T).T

        Tx = self.prox_func(z, tau * self.gamma)  # new

        return Tx

    def forward(self, d, **kwargs):
        K = kwargs.get('K')

        xk = np.zeros([d.shape[0], self.n])
        for i in range(K):
            xk = self.T(xk, d, index=i)

            self.iter_history.append(xk)
        return xk

    def model_name(self) -> str:
        return 'ISTA'

    def proximal_operator_name(self) -> str:
        print(self.prox_func.__name__)
        return self.prox_func.__name__

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
