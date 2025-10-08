from abc import ABCMeta
from typing import Optional

import numpy as np

__all__ = ['ProximalOperator', 'GroupProximalOperator']


class ProximalOperator(metaclass=ABCMeta):
    latex_name: Optional[str]  # the latex name of the proximal operator

    def calculate_subderivative(self, u: np.ndarray, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def obj(cls, x: np.ndarray, xtilde: np.ndarray, nu: float) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def prox(cls, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        """ calculate the proximal operator

           :param x: np.ndarray, the input vector(matrix)
           :param v: T, the threshold
           :param args: None        :param kwargs: None
           :return: np.ndarray, the proximal operator result
        """
        raise NotImplementedError

    @classmethod
    def name(cls):
        return cls.__name__

    def __call__(self, *args, **kwargs):
        return self.prox(*args, **kwargs)

    def __str__(self):
        pass


class GroupProximalOperator(ProximalOperator):
    n: int  # the dimension
    gLen: int  # the group length
    num_subvectors: int  # the number of subvectors

    latex_name: Optional[str]   # the latex name of the proximal operator

    __slots__ = ('n', 'gLen', 'num_subvectors', 'num_samples', 'latex_name')

    @staticmethod
    def get_num_subvectors(n: int, gLen: int) -> int:
        return n // gLen

    @classmethod
    def obj(cls, x: np.ndarray, xtilde: np.ndarray, nu: float) -> np.ndarray:
        """
        calculate the objective function, i.e.  J_\ell_1(x)
            obj=J(x)=nu*||x||_1+0.5*||x-xtilde||_2^2
        :param x:
        :param xtilde:
        :param nu:
        :return:
        """
        raise NotImplementedError

    def _split_vectors(self, x: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
