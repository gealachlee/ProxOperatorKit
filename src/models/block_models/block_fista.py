# ** coding: utf-8 **
"""
@Author: Zhihong Li, Rongrong Lin
@date: 2020/11/25
@description: FISTA.
@version: 2.0
"""
import numpy as np
from models.base import Model
from typing import Callable

__all__ = ['FISTA']


class FISTA(Model):

    def __init__(self, A: np.ndarray, tau, prox_func: Callable):
        super().__init__(A, tau)
        self.m, self.n = self.A.shape
        self.L_np = np.linalg.norm(np.matmul(A.transpose(), A), ord=2)
        self.gamma = 1 / np.linalg.norm(A, 2) ** 2
        self.prox_func: Callable = prox_func
        self.iter_history: list = []

    def name(self):
        return f'FISTA with {self.prox_func.name()}'

    def T(self, x, d, **kwargs):
        tau = kwargs.get('tau', self.tau)
        index = kwargs.get('index', -1)
        assert index >= 0

        r = (self.A @ x.T).T - d
        z = x - self.gamma * 2 * (self.A.T @ r.T).T

        Tx = self.prox_func(z, self.gamma * tau)
        #  Tx = self.prox_func(Tx, self.gamma * tau)

        return Tx

    def forward(self, d, **kwargs) -> np.ndarray:
        K = kwargs.get('K')

        xk = z = x_next = np.zeros([d.shape[0], self.n])

        tk = t_next = 1.0

        for i in range(K):
            # Process momentum

            x_next = self.T(z, d, index=i)
            t_next = 0.5 + np.sqrt(1.0 + 4.0 * tk ** 2) / 2.0
            z_next = xk + ((tk - 1.0) / t_next) * (x_next - xk)
            # print((tk -1.0)/t_next)

            # Process iteration
            xk = x_next
            z = z_next
            tk = t_next

            self.iter_history.append(xk)
        return xk

    def model_name(self) -> str:
        return 'FISTA'

    def proximal_operator_name(self) -> str:
        print(self.prox_func.__name__)
        return self.prox_func.__name__

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
