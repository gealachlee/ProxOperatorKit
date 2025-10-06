from typing import Callable

import numpy as np

from models.base import Model
from models.block_models import ISTA

class JointISTA(Model):
    def __init__(self, A: np.ndarray, tau,
                 prox_func1: Callable,
                 prox_func2: Callable):
        super().__init__(A, tau)
        self.tau = tau
        self.prox_func1 = prox_func1
        self.prox_func2 = prox_func2
        self.m, self.n = self.A.shape
        self.L_np = np.linalg.norm(np.matmul(A.transpose(), A), ord=2)
        self.gamma1 = 1 / np.linalg.norm(A, 2) ** 2
        self.gamma2 = 1 - self.gamma1

    def name(self):
        return f'ISTA with joint sparse {self.prox_func1.name()} and {self.prox_func2.name()}'

    def T(self, x, d, **kwargs):
        tau = kwargs.get('tau', self.tau)
        index = kwargs.get('index', -1)
        assert index >= 0

        r = (self.A @ x.T).T - d
        z = x - self.gamma1 * 2 * (self.A.T @ r.T).T

        Tx = self.prox_func2(self.prox_func1(z, 0.05* self.gamma1 * tau), 0.95* self.gamma1 * tau)
        return Tx

    def model_name(self) -> str:
        return 'JointISTA'

    def proximal_operator_name(self) -> str:
        return f'{self.prox_func1} {self.prox_func2}'

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
