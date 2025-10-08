# ** coding: utf-8 **
"""
@Author: Zhihong Li, Rongrong Lin
@date: 2020/11/25
@description: FISTA.
@version: 2.0
"""
from typing import Union

import numpy as np

from models.base import GroupModel
from prox import GroupProximalOperator,ProximalOperator

__all__ = ['GROUPPGAC']


class GROUPPGAC(GroupModel):
    """
    proximal gradient algorithm with continuation technique in group (block) scene
    reference: Y. Hu, X. Hu, C. K. W. Yu and J. Qin, Joint sparse optimization: lower-order regularization method and
    application in cell fate conversion, Inverse Probl. 40(9) (2024) p. 095003.
    """

    def __init__(self, A: np.ndarray, tau, prox_func: Union[GroupProximalOperator, ProximalOperator], **kwargs):
        super().__init__(A, tau, prox_func)
        self.m, self.n = self.A.shape
        self.L_np = np.linalg.norm(np.matmul(A.transpose(), A), ord=2)
        self.min_gamma1 = 1e-4  # DEFAULT EXPERIMENT1 1E-4
        self.min_gamma2 = 1e-6

        self.stepsize = 0.1  # 0.09 # DEFAULT EXPERIMENT1 0.1
        self.desc_rate = 0.99
        self.prox_func: GroupProximalOperator = prox_func
        self.gamma1 =0.5  # DEFAULT 1
        self.iter_history: list = []

        for k, v in kwargs.items():
            setattr(self, k, v)

    def T(self, x, d, **kwargs):
        tau = kwargs.get('tau', self.tau)
        index = kwargs.get('index', -1)
        assert index >= 0

        r = (self.A @ x.T).T - d
        z = x - self.stepsize * 2 * (self.A.T @ r.T).T
        Tx = self.prox_func(z, (self.stepsize * 2) * np.maximum(self.gamma1, self.min_gamma1))  # ())
        return Tx

    def forward(self, d, **kwargs) -> np.ndarray:
        K = kwargs.get('K')

        xk = np.zeros([d.shape[0], self.n])

        for i in range(K):
            xk = self.T(xk, d, index=i)
            self.iter_history.append(xk)
            self.gamma1 = self.desc_rate * self.gamma1

        return xk

    def proximal_operator_name(self) -> str:

        return self.prox_func.__name__
