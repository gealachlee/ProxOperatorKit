# ** coding: utf-8 **
"""
@Author: Zhihong Li
@date: 2024/09/25
@description: models for the project.
@version: 3.0
"""

from abc import ABCMeta

import numpy as np

__all__ = ['Model', 'JointModel', 'GroupModel']

from prox import GroupProximalOperator, ProximalOperator


class Model(metaclass=ABCMeta):
    iter_history: list[np.ndarray] = []

    def __init__(self, A, tau, **kwargs):
        self.A = A  # observed metric
        self.tau = tau  # Parameter for problem definition
        for k, v in kwargs.items():
            setattr(self, k, v)

    def forward(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def desc(self) -> str:
        raise NotImplementedError

    def proximal_operator_name(self) -> str:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class GroupModel(Model):
    def __init__(self, A: np.ndarray, tau, prox_func: GroupProximalOperator):
        super().__init__(A, tau)
        self.m, self.n = self.A.shape
        self.L_np = np.linalg.norm(np.matmul(A.transpose(), A), ord=2)
        self.gamma = 1 / np.linalg.norm(A, 2) ** 2
        self.prox_func: GroupProximalOperator = prox_func
        self.iter_history: list = []

    def desc(self) -> str:
        return f'Algorithm: {self.__class__.__name__} with proxmal operator:{self.prox_func.name()}'

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class JointModel(Model):
    def __init__(self, A: np.ndarray, tau, prox_func1: ProximalOperator, prox_func2: GroupProximalOperator):
        super().__init__(A, tau)
        self.m, self.n = self.A.shape
        self.L_np = np.linalg.norm(np.matmul(A.transpose(), A), ord=2)
        self.prox_func1: ProximalOperator = prox_func1
        self.prox_func2: GroupProximalOperator = prox_func2
        self.iter_history: list = []

    def desc(self) -> str:
        return f'Algorithm: {self.__class__.__name__} with proxmal operator {self.prox_func1.name()} and {self.prox_func2.name()}'

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
