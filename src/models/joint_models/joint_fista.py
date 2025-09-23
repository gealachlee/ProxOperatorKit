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
from prox import ProximalOperator

__all__ = ['JointFISTA']


class JointFISTA(Model):

    def __init__(self, A: np.ndarray, tau, prox_func1: ProximalOperator, prox_func2: ProximalOperator):
        super().__init__(A, tau)
        self.m, self.n = self.A.shape
        self.L_np = np.linalg.norm(np.matmul(A.transpose(), A), ord=2)
        self.gamma1 = 1 / np.linalg.norm(A, 2) ** 2
        self.gamma2 = 1 - self.gamma1
        self.prox_func1: ProximalOperator = prox_func1
        self.prox_func2: ProximalOperator = prox_func2
        self.iter_history: list = []

    def name(self):
        return f'JointFISTA with {self.prox_func1.name()} and {self.prox_func2.name()}'

    def T(self, x, d, **kwargs):
        tau = kwargs.get('tau', self.tau)
        index = kwargs.get('index', -1)
        assert index >= 0

        r = (self.A @ x.T).T - d
        # z = x - self.gamma1 * 2 * (self.A.T @ r.T).T
        z = x - self.gamma1 * 2 * (self.A.T @ r.T).T
        z = self.prox_func1(z, 0.05* self.gamma1 * tau)  # 0.001*self.gamma2 * tau)
        Tx = self.prox_func2(z, 0.95 * self.gamma1 * tau)  # self.gamma1 * tau)  # new
        # z = self.prox_func1(z,self.gamma1 * tau)
        # Tx = self.prox_func2(z, self.gamma1 * tau)  # new
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
        return 'JointFISTA'

    def proximal_operator_name(self) -> str:
        print(self.prox_func.__name__)
        return self.prox_func.__name__

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
