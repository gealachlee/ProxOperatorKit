# ** coding: utf-8 **
"""
@Author: Zhihong Li, Rongrong Lin
@date: 2024/12/19
@description: Proximal operators for L1\psi
@version: 1.0
"""
from typing import Union

import numpy as np
from common import T
from prox import GroupProximalOperator
from prox.sep_prox.prox_cl import *
from abc import ABCMeta
from prox import ProximalOperator


class L1_SubvecProx(metaclass=ABCMeta):
    # con1: T = None
    # con2: T = None
    # con4: T = None
    precompute: bool
    #  s_arange = None
    #  gLen = None
    #  element_prox = None
    #  zeros = None
    #  lamb = None

    __slots__ = ['con1', 'con2', 'con4', 'precompute', 's_arange', 'gLen', 'element_prox', 'zeros', 'lamb']

    def __init__(self, gLen, lamb: T, element_prox: ProximalOperator):
        self.s_arange = None
        self.gLen = gLen
        self.lamb = lamb
        self.element_prox = element_prox
        self.zeros = np.zeros(gLen)
        self.s_arange = np.arange(1, gLen + 1)
        self.con1 = None

    def calculate_con1(self, *args, **kwargs) -> Union[T, int, float, np.ndarray]:
        raise NotImplementedError

    def calculate_con2(self, *args, **kwargs) -> Union[T, int, float, np.ndarray]:
        raise NotImplementedError

    def calculate_con4(self, *args, **kwargs) -> Union[T, int, float, np.ndarray]:
        raise NotImplementedError

    def obj(self, y: np.ndarray, ytilde: np.ndarray, lamb: T) -> np.ndarray:
        """
        Object function for L1\psi, such as : lamb*\|y\|_1+0.5*\|y-ytilde\|_2^2
        :param y:
        :param ytilde:
        :param lamb:
        :return:
        """
        raise NotImplementedError

    def prox(self, x, v):

        con1, con2, con4 = self.compute_conditions(v)
        self.lamb = v

        x_abs = np.abs(x)
        ys_l1 = x_abs.sum()

        # if ys_l1 <= con1:
        #     return self.zeros
        if np.linalg.norm(x, ord=2) <= con1:
            return self.zeros
        else:
            sign_x = np.sign(x)
            indices = np.argsort(-x_abs)
            y = x_abs[indices]

            if y[-1] > con2:
                return (x_abs - self._calculate_cs(y_l1_array=ys_l1, nu=v, s=len(x))) * sign_x
            else:

                sorted_indices = np.argsort(indices)
                s_list = [self.zeros]
                s_list.extend(self._loop_search_s(y, v, con4))
                if len(s_list)>1:
                    s_list.pop(0)
                    Js_list = [self.obj(y, per_ytilde, v) for per_ytilde in s_list]
                else:

                    Js_list = [self.obj(y, per_ytilde, v) for per_ytilde in s_list]
                return self._rearrange_org(s_list[np.argmin(Js_list)], sorted_indices, sign_x)

    def name(self) -> str:
        return self.element_prox.name()

    @staticmethod
    def _rearrange_org(ytilde: np.ndarray, sorted_indices, sign: np.ndarray) -> np.ndarray:
        return ytilde[sorted_indices] * sign

    def _calculate_cs(self, y_l1_array: np.ndarray, nu: np.ndarray, s) -> np.ndarray:
        return self.lamb * (self.element_prox.calculate_subderivative(
            self.element_prox.prox(y_l1_array, v=nu * s))
        )

    def _loop_search_s(self, y, nu, con4):
        y_l1_array = y.cumsum()
        # s_arange = np.arange(1, len(y) + 1)
        cs_array = self._calculate_cs(y_l1_array, nu, self.s_arange)  # y -> y_s
        y_plus = np.concatenate((y[1:], [-1]))
        res = (y_l1_array > con4) & (y > cs_array) & (cs_array >= y_plus)
        return [np.maximum(y - cs_array[ind], 0) for ind in np.where(res)[0]]

    def compute_conditions(self, v):
        con1 = self.calculate_con1(v)
        con2 = self.calculate_con2(v)
        con4 = self.calculate_con4(v, self.s_arange)
        return con1, con2, con4


class GeneralProxL1Psi(GroupProximalOperator):

    def __init__(self, n: int, gLen: int, subvec_prox: L1_SubvecProx, num_samples: int = 1):
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)
        self.subvec_prox: L1_SubvecProx = subvec_prox
        self.num_samples: int = num_samples

    def prox(self, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        x_subvectors = self._split_vectors(x)
        subvec_prox = self.subvec_prox


        # res = np.apply_along_axis(np.count_nonzero, 2, lp) #计算l0范数

        # print(res)
        return self._concentrate_vectors([subvec_prox.prox(x_sub,v) for x_sub in x_subvectors]).reshape(
            self.num_samples, -1)
        # return self._concentrate_vectors(
        #     [subvec_prox.prox(x_sub, r) for r, x_sub in zip(v, x_subvectors)]).reshape(
        #     self.num_samples, -1)

    def name(self):
        return self.subvec_prox.name()

    def _split_vectors(self, x: np.ndarray) -> list[np.ndarray]:
        return np.split(x.flatten(), self.num_subvectors * self.num_samples)

    @classmethod
    def _concentrate_vectors(self, x_subvectors: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(x_subvectors)


class GeneralProxL2Psi(GroupProximalOperator):

    def __init__(self, n: int, gLen: int, subvec_prox: ProximalOperator):
        """
        初始化函数，用于创建一个新的实例。

        参数:
            n (int): 向量的总长度。
            gLen (int): 每个子向量的长度。
            subvec_prox (ProximalOperator): 用于子向量的近端算子实例。

        属性:
            n (int): 存储向量的总长度。
            gLen (int): 存储每个子向量的长度。
            num_subvectors (int): 计算并存储子向量的数量。
            subvec_prox (ProximalOperator): 存储用于子向量的近端算子实例。
        """
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)
        self.subvec_prox: ProximalOperator = subvec_prox


    def prox(self, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        x_subvectors = self._split_vectors(x)

        norms = np.stack([np.linalg.norm(subvec, ord=2, axis=1) for subvec in x_subvectors], axis=0)

        tao = self.subvec_prox.prox(norms, v) / (norms)
        x_subvectors = [tao[i].reshape(-1, 1) * x_subvectors[i] for i in range(self.num_subvectors)]
        return self._concentrate_vectors(x_subvectors)

    def name(self):
        return self.subvec_prox.name()

    def _split_vectors(self, x: np.ndarray) -> list[np.ndarray]:
        return np.split(x, self.num_subvectors, axis=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(x_subvectors, axis=1)
