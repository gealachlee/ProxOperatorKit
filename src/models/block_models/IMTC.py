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
from prox.group_prox import ProxL2_0
from prox.sep_prox.prox_cl import ProxCapped1over2, ProxSCAD
from prox import GroupProximalOperator

__all__ = ['GROUPIMTC']

from prox.group_prox.general import GeneralProxL2Psi


class GROUPIMTC(Model):

    def __init__(self, A: np.ndarray, tau, prox_func: GroupProximalOperator):
        super().__init__(A, tau)
        self.m, self.n = self.A.shape
        self.L_np = np.linalg.norm(np.matmul(A.transpose(), A), ord=2)
        self.min_gamma1 = 1e-4  # DEFAULT EXPERIMENT1 1E-4
        self.min_gamma2 = 1e-6

        self.stepsize = 0.09 # 0.09 # DEFAULT EXPERIMENT1 0.1
        self.desc_rate = 0.99
        self.prox_func: GroupProximalOperator = prox_func
        self.gamma1 =1 # DEFAULT 1

        # TL1
        # self.stepsize = 0.09  # 0.09 # DEFAULT EXPERIMENT1 0.1
        # self.desc_rate = 0.996  # 94  # DEFAULT EXPERIMENT1 0.994
        # self.prox_func: GroupProximalOperator = prox_func
        # self.gamma1 = 0.8  # DEFAULT 1

        # try:
        #     if prox_func.subvec_prox.name() == 'Tl1':
        #         print('\n----------\n')
        #         self.desc_rate = 0.99
        #         self.stepsize =  0.01#0.08  # 0.18
        #         self.gamma1 = 2
        #         self.min_gamma1 = 1e-4
        # except:
        #     pass

        #suitable
        # self.stepsize = 0.09  # 0.09 # DEFAULT EXPERIMENT1 0.1
        # self.desc_rate = 0.998

        if isinstance(self.prox_func, (ProxL2_0)):  # 对于Cap1/2, gamma 偏小
            self.desc_rate = 0.994
            self.gamma1 = 3
        if isinstance(self.prox_func, ProxCapped1over2):  # 对于Cap1/2, gamma 偏小
            self.desc_rate = 0.98

            # if isinstance(self.prox_func.subvec_prox, ProxCapped1over2):
            #     self.gamma1 = 1  # 0.25**2/(2*self.stepsize)# DEFAULT 1
        self.iter_history: list = []

    def name(self):
        return f'group IMTC with {self.prox_func.name()} '

    def T(self, x, d, **kwargs):
        tau = kwargs.get('tau', self.tau)
        index = kwargs.get('index', -1)
        assert index >= 0

        r = (self.A @ x.T).T - d
        # grad_norm = np.linalg.norm(2 * (self.A.T @ r.T).T)
        # self.stepsize = min(self.stepsize , 1.0 / grad_norm)

        z = x - self.stepsize * 2 * (self.A.T @ r.T).T

      #  res = np.apply_along_axis(np.count_nonzero, 1, x_subvectors)
        Tx = self.prox_func(z, (self.stepsize * 2) * np.maximum(self.gamma1, self.min_gamma1))  # ())
        return Tx

    def forward(self, d, **kwargs) -> np.ndarray:
        K = kwargs.get('K')

        xk = np.zeros([d.shape[0], self.n])
        # self.gamma1 = 1

        for i in range(K):
            xk = self.T(xk, d, index=i)
            self.iter_history.append(xk)
            self.gamma1 = self.desc_rate * self.gamma1


        return xk

    def model_name(self) -> str:
        return 'GROUPIMTC'

    def proximal_operator_name(self) -> str:
        print(self.prox_func.__name__)
        return self.prox_func.__name__

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
