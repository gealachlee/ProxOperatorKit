# ** coding: utf-8 **
"""
@Author: Zhihong Li, Rongrong Lin
@date: 2024/11/25
@description: IMTC.
@version: 2.0
"""
import numpy as np
from models.base import JointModel
from prox import ProximalOperator, GroupProximalOperator

__all__ = ['PGAC']

from prox.group_prox import ProxL2_0


class PGAC(JointModel):
    """
     proximal gradient algorithm with continuation technique in joint scene.
     reference: Y. Hu, X. Hu, C. K. W. Yu and J. Qin, Joint sparse optimization: lower-order regularization method and
     application in cell fate conversion, Inverse Probl. 40(9) (2024) p. 095003.
     """
    def __init__(self, A: np.ndarray, tau, prox_func1: ProximalOperator, prox_func2: GroupProximalOperator,**kwargs):
        super().__init__(A, tau, prox_func1, prox_func2)
        self.m, self.n = self.A.shape
        self.L_np = np.linalg.norm(np.matmul(A.transpose(), A), ord=2)
        self.min_gamma1 = 1e-4
        self.min_gamma2 = 1e-6
        self.stepsize = 0.09
        self.desc_rate = 0.994
        self.gamma1 = 0.5  # 1 - self.gamma2
        self.gamma2 = 0.05  # 1 - self.gamma1
        self.prox_func1: ProximalOperator = prox_func1  #
        self.prox_func2: GroupProximalOperator = prox_func2
        self.iter_history: list = []
        if isinstance(self.prox_func2, ProxL2_0):  # 对于Cap1/2, gamma 偏小
            self.gamma1 = 0.1
            self.gamma2 = 0.1
            self.desc_rate = 0.994
            self.min_gamma2 = 1e-6

        for k, v in kwargs.items():
            setattr(self, k, v)

    def name(self):
        return f'{self.__name__} with {self.prox_func1.name()} and {self.prox_func2.name()}'

    def T(self, x, d, **kwargs):
        tau = kwargs.get('tau', self.tau)
        index = kwargs.get('index', -1)
        assert index >= 0

        r = (self.A @ x.T).T - d
        z = x - self.stepsize * 2 * (self.A.T @ r.T).T
        z = self.prox_func1(z,  self.stepsize*2* (np.maximum(self.gamma2, self.min_gamma2)))
        lp = self.prox_func2._split_vectors(z)

        if 'L0'==self.prox_func1.name():
            res = np.apply_along_axis(np.count_nonzero, 2, lp)

        else:
            lamb_func = lambda r: np.sum(np.abs(r) ** float(self.prox_func1.name()[-1]))
            res= np.apply_along_axis(lamb_func,2,lp)
        Tx = self.prox_func2(z,
                             self.stepsize * 2 * (np.maximum(self.gamma1, self.min_gamma1)
                                                  +
                                                  np.maximum(self.gamma2, self.min_gamma2) * res)
                             )
        return Tx

    def forward(self, d, **kwargs):
        K = kwargs.get('K')

        xk = np.zeros([d.shape[0], self.n])
        for i in range(K):
            xk = self.T(xk, d, index=i)

            self.iter_history.append(xk)
            self.gamma1 = self.desc_rate * self.gamma1
            self.gamma2 = self.desc_rate * self.gamma2
        return xk
