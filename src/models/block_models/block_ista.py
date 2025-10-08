# ** coding: utf-8 **
"""
@Author: Zhihong Li, Rongrong Lin
@date: 2024/11/25
@description: ISTA.
@version: 2.0
"""
from typing import Callable

import numpy as np

from models.base import Model, GroupModel
from prox import GroupProximalOperator,ProximalOperator
from typing import  Union


class ISTA(GroupModel):

    def __init__(self, A: np.ndarray, tau, prox_func: Union[GroupProximalOperator,ProximalOperator]):
        super().__init__(A, tau, prox_func)
        self.m, self.n = self.A.shape
        self.L_np = np.linalg.norm(np.matmul(A.transpose(), A), ord=2)
        self.gamma = 1 / np.linalg.norm(A, 2) ** 2 # 步长
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


    def proximal_operator_name(self) -> str:
        return self.prox_func.__name__


