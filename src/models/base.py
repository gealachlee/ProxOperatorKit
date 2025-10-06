# ** coding: utf-8 **
"""
@Author: Zhihong Li
@date: 2023/11/25
@description: models for the project.
@version: 2.0
"""

from abc import ABCMeta

import numpy as np

__all__ = ['Model']


class Model(metaclass=ABCMeta):
    iter_history: list[np.ndarray] = []

    def __init__(self, A, tau, **kwargs):
        self.A = A  # observed metric
        self.tau = tau  # Parameter for problem definition
        for k, v in kwargs.items():
            setattr(self, k, v)

    def forward(self,*args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def model_name(self) -> str:
        raise NotImplementedError

    def proximal_operator_name(self) -> str:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
