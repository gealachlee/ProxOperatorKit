# ** coding: utf-8 **
"""
@Author: Zhihong Li, Rongrong Lin
@date: 2024/11/25
@description: Proximal operators for L2q and L1q
@version: 2.0
"""
import numpy as np

from common import T
from prox import GroupProximalOperator
from prox.sep_prox.prox_cl import *
import time

__all__ = [
    'ProxL1_1over2',
    'ProxL1_2over3',
    'ProxL1_1over2_FixParams',
    'ProxL1_2over3_FixParams'
]


# class ProxL1_1over2(GroupProximalOperator):
#     """
#     Standard implementation of the proximal operator for the \ell_{1,\frac{1}{2}}.
#     Ralated Article:
#     [1] Lin R, Chen S, Feng H, et al. Computing the Proximal Operator of the $\ell_ {1, q} $-norm for Group Sparsity[J]. arXiv preprint arXiv:2409.14156, 2024.
#     """
#
#     def __init__(self, n: int, gLen: int, num_samples: int = 1):
#         self.n: int = n
#         self.gLen: int = gLen
#         self.num_subvectors: int = self.get_num_subvectors(n, gLen)
#         self.num_samples: int = num_samples
#
#     @classmethod
#     def obj(cls, x: np.ndarray, xtilde: np.ndarray, nu: T) -> np.ndarray:
#         return super().obj(x, xtilde, nu)
#
#     def prox(self, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
#         x_subvectors = self._split_vectors(x)
#         return np.concatenate([self.L1_1over2_part.prox(x_sub, v) for x_sub in x_subvectors]).reshape(self.num_samples,
#                                                                                                       -1)
#
#     @classmethod
#     def name(cls):
#         return 'group_l1_1over2'
#
#     def _split_vectors(self, x: np.ndarray) -> list[np.ndarray]:
#         return np.split(x.flatten(), self.num_subvectors * self.num_samples)
#
#     @staticmethod
#     def _concentrate_vectors(x_subvectors: list[np.ndarray]) -> np.ndarray:
#         raise NotImplementedError('You dont need to call this method')
#
#     class L1_1over2_part:
#
#         @staticmethod
#         def obj(y: np.ndarray, ytilde: np.ndarray, nu: T) -> np.ndarray:
#             return nu * np.power(ytilde.sum(), 1 / 2) + 0.5 * (np.linalg.norm(y - ytilde, 2) ** 2)
#
#         @classmethod
#         def prox(cls, x: np.ndarray, nu: T) -> np.ndarray:
#             # 记录顺序和绝对值，用于后续计算
#
#             x_abs = np.abs(x)
#
#             condition_1 = 3 * nu ** (2 / 3) / (2 ** (4 / 3))
#
#             ys_l1 = x_abs.sum()
#             # ys_l1 = y.sum()
#             if ys_l1 <= condition_1:  # case1
#                 return np.zeros_like(x)
#             else:
#                 condition_2 = 1.5 * nu ** (2 / 3)
#                 sign_x = np.sign(x)
#
#                 # if y[-1] > condition_2:  # case2
#                 indices = np.argsort(-x_abs)
#                 y = x_abs[indices]
#                 if y[-1] > condition_2:  # case2
#                     return (x_abs - cls._calculate_cs(ys_l1, nu, len(x))) * sign_x
#                 else:
#                     sorted_indices = np.argsort(indices)
#                     # Search s
#                     s_list = [np.zeros_like(y)]
#                     s_list.extend(cls._loop_search_s(y, nu))
#                     Js_list = [cls.obj(y, per_ytilde, nu) for per_ytilde in s_list]
#                     if np.nan in Js_list:
#                         raise ValueError('NaN in Js_list')
#                     if y[0] <= condition_2:  # case 3
#                         return cls._rearrange_org(s_list[np.argmin(Js_list)], sorted_indices, sign_x)
#                     # return cls._rearrange_org(s_list[1:][np.argmin(Js_list[1:])], sorted_indices, sign_x)
#                     return cls._rearrange_org(s_list[np.argmin(Js_list)], sorted_indices, sign_x)
#                     # return cls._rearrange_org(s_list[-1], sorted_indices, sign_x)
#
#         @staticmethod
#         def _rearrange_org(ytilde: np.ndarray, sorted_indices, sign: np.ndarray) -> np.ndarray:
#             return ytilde[sorted_indices] * sign
#
#         @classmethod
#         def _loop_search_s(cls, y: np.ndarray, nu: np.ndarray) -> list[np.ndarray]:
#             # Search s
#             y_l1_array = y.cumsum()
#
#             s_arange = np.arange(1, len(y_l1_array) + 1)
#             cs_array = cls._calculate_cs(y_l1_array, nu, s_arange)  # y -> y_s
#             y_plus = np.append(y, -1)[1:]
#
#             condition4 = (((3 * nu ** (2 / 3)) / (2 ** (4 / 3)))) * s_arange ** (2 / 3)
#
#             res = (y_l1_array >= condition4) & (
#                     y > cs_array) & (cs_array >= y_plus)
#             return [np.maximum(y - cs_array[-1], 0)] if len(np.where(res)[0]) > 0 else []
#
#         @staticmethod
#         def _calculate_cs(y_l1_array: np.ndarray, nu: np.ndarray, s) -> np.ndarray:
#             return (np.sqrt(3.0) * nu) / (
#                     4 * np.sqrt(y_l1_array) * np.cos(
#                 (1 / 3) * np.arccos(-3 * np.sqrt(3.0) * nu * s * np.power(y_l1_array, -3 / 2) / 4)))
#
#         @staticmethod
#         def _calculate_condition3(v, q, s):  # cs
#             return (2 - q) * (v * q * s / ((q - 1) ** (1 - q))) ** (1 / (2 - q))
#
#
# class ProxL1_2over3(GroupProximalOperator):
#     """
#     Standard implementation of the proximal operator for the \ell_{1,\frac{2}{3}}.
#     Ralated Article:
#     [1] Lin R, Chen S, Feng H, et al. Computing the Proximal Operator of the $\ell_ {1, q} $-norm for Group Sparsity[J]. arXiv preprint arXiv:2409.14156, 2024.
#
#     """
#
#     def __init__(self, n: int, gLen: int, num_samples: int = 1):
#         self.n: int = n
#         self.gLen: int = gLen
#         self.num_subvectors: int = self.get_num_subvectors(n, gLen)
#         self.num_samples: int = num_samples
#
#     @classmethod
#     def obj(cls, x: np.ndarray, xtilde: np.ndarray, nu: T) -> np.ndarray:
#         return super().obj(x, xtilde, nu)
#
#     def prox(self, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
#         x_subvectors = self._split_vectors(x)
#         return np.concatenate([self.L1_2over3_part.prox(x_sub, v) for x_sub in x_subvectors]).reshape(self.num_samples,
#                                                                                                       -1)
#
#     @classmethod
#     def name(cls):
#         return 'group_l1_2over3'
#
#     def _split_vectors(self, x: np.ndarray) -> list[np.ndarray]:
#         return np.split(x.flatten(), self.num_subvectors * self.num_samples)
#
#     @staticmethod
#     def _concentrate_vectors(x_subvectors: list[np.ndarray]) -> np.ndarray:
#         raise NotImplementedError('You dont need to call this method')
#
#     class L1_2over3_part():
#
#         @staticmethod
#         def obj(y: np.ndarray, ytilde: np.ndarray, nu: T) -> np.ndarray:
#             return nu * np.power(ytilde.sum(), 2 / 3) + 0.5 * (np.linalg.norm(y - ytilde, 2) ** 2)
#
#         @classmethod
#         def prox(cls, x: np.ndarray, nu: T) -> np.ndarray:
#             x_abs = np.abs(x)
#             condition_1 = 4 * ((2 / 9) ** 0.75) * (nu ** 0.75)
#             ys_l1 = x_abs.sum()
#
#             if ys_l1 <= condition_1:  # case1
#                 return np.zeros_like(x)
#             else:
#                 condition_2 = 2 * ((2 * nu / 3) ** 0.75)
#                 sign_x = np.sign(x)
#                 indices = np.argsort(-x_abs)
#                 y = x_abs[indices]
#                 if y[-1] > condition_2:
#                     return (x_abs - cls._calculate_cs(ys_l1, nu, len(x))) * sign_x
#                 else:
#                     # Search s
#                     sorted_indices = np.argsort(indices)
#                     s_list = [np.zeros_like(y)]
#                     s_list.extend(cls._loop_search_s(y, nu))
#                     Js_list = [cls.obj(y, per_ytilde, nu) for per_ytilde in s_list]
#                     return cls._rearrange_org(s_list[np.argmin(Js_list)], sorted_indices, sign_x)
#
#         @staticmethod
#         def _rearrange_org(ytilde: np.ndarray, sorted_indices, sign: np.ndarray) -> np.ndarray:
#             return ytilde[sorted_indices] * sign
#
#         @classmethod
#         def _loop_search_s(cls, y: np.ndarray, nu: np.ndarray) -> list[np.ndarray]:
#             # Search s
#             y_l1_array = y.cumsum()
#
#             s_arange = np.arange(1, len(y_l1_array) + 1)
#             cs_array = cls._calculate_cs(y_l1_array, nu, s_arange)
#             y_plus = np.append(y, -1)[1:]
#             if y.sum() < 1e-5:
#                 return [np.maximum(y - cs_array[ind], 0) for ind in np.where(y > cs_array)[0]]
#             condition4 = ((4 / 3) * nu ** 0.75 * ((2 ** 0.75) / (3 ** 0.5))) * s_arange ** 0.75
#             res = (y_l1_array >= condition4) & (
#                     y > cs_array) & (
#                           cs_array >= y_plus)  # & (y_l1_array >= cls._calculate_condition3(v=nu, q=2 / 3, s=s_arange))
#             return [np.maximum(y - cs_array[ind], 0) for ind in np.where(res)[0]]
#
#         @staticmethod
#         def _calculate_condition3(v, q, s):
#             return (2 - q) * (v * q * s / ((q - 1) ** (1 - q))) ** (1 / (2 - q))
#
#         @staticmethod
#         def _calculate_cs(y_l1_array: np.ndarray, nu: np.ndarray, s) -> np.ndarray:
#             tmp = y_l1_array ** 4 / 256 - 8 * (nu ** 3) * (s ** 3) / 729
#             part_alpha = np.where(tmp < 0, np.sqrt(np.sqrt(8 * (nu ** 3) * (s ** 3) / 729)), np.sqrt(tmp))
#
#             alpha_s = (y_l1_array ** 2 / 16 + part_alpha) ** (1 / 3) + (np.abs(y_l1_array ** 2 / 16 - part_alpha)) ** (
#                     1 / 3)
#             sqrt_2alpha = np.sqrt(2 * alpha_s)
#             return (4.0 * nu / 3.0) / (sqrt_2alpha + np.sqrt(np.abs(2 * y_l1_array / sqrt_2alpha - 2 * alpha_s)))



class ProxL1_1over2(GroupProximalOperator):
    """
    Standard implementation of the proximal operator for the \ell_{1,\frac{1}{2}}.
    Ralated Article:
    [1] Lin R, Chen S, Feng H, et al. Computing the Proximal Operator of the $\ell_ {1, q} $-norm for Group Sparsity[J]. arXiv preprint arXiv:2409.14156, 2024.
    """

    def __init__(self, n: int, gLen: int, num_samples: int = 1):
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)
        self.num_samples: int = num_samples

    @classmethod
    def obj(cls, x: np.ndarray, xtilde: np.ndarray, nu: T) -> np.ndarray:
        return super().obj(x, xtilde, nu)

    def prox(self, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        x_subvectors = self._split_vectors(x)
        return np.concatenate([self.L1_1over2_part.prox(x_sub, v) for x_sub in x_subvectors]).reshape(self.num_samples,
                                                                                                      -1)

    @classmethod
    def name(cls):
        return 'group_l1_1over2'

    def _split_vectors(self, x: np.ndarray) -> list[np.ndarray]:
        return np.split(x.flatten(), self.num_subvectors * self.num_samples)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError('You dont need to call this method')

    class L1_1over2_part:

        @staticmethod
        def obj(y: np.ndarray, ytilde: np.ndarray, nu: T) -> np.ndarray:
            return nu * np.power(ytilde.sum(), 1 / 2) + 0.5 * (np.linalg.norm(y - ytilde, 2) ** 2)

        @classmethod
        def prox(cls, x: np.ndarray, nu: T) -> np.ndarray:
            # 记录顺序和绝对值，用于后续计算

            x_abs = np.abs(x)

            condition_1 = 3 * nu ** (2 / 3) / (2 ** (4 / 3))

            ys_l1 = x_abs.sum()
            # ys_l1 = y.sum()
            if ys_l1 <= condition_1:  # case1
                return np.zeros_like(x)
            else:
                condition_2 = 1.5 * nu ** (2 / 3)
                sign_x = np.sign(x)

                # if y[-1] > condition_2:  # case2
                indices = np.argsort(-x_abs)
                y = x_abs[indices]
                if y[-1] > condition_2:  # case2
                    # Method 1
                    # ytilde = y - cls._calculate_cs(ys_l1, nu, len(y))
                    # return #cls._rearrange_org(ytilde, sorted_indices, sign_x)  # 先返回原始顺序，再去除绝对值
                    # Method 2
                    return (x_abs - cls._calculate_cs(ys_l1, nu, len(x))) * sign_x
                else:
                    sorted_indices = np.argsort(indices)
                    # Search s
                    s_list = [np.zeros_like(y)]
                    s_list.extend(cls._loop_search_s(y, nu))
                    Js_list = [cls.obj(y, per_ytilde, nu) for per_ytilde in s_list]
                    # if y[0] <= condition_2:  # case 3
                    #     return cls._rearrange_org(s_list[np.argmin(Js_list)], sorted_indices, sign_x)
                    # return cls._rearrange_org(s_list[1:][np.argmin(Js_list[1:])], sorted_indices, sign_x)
                    return cls._rearrange_org(s_list[np.argmin(Js_list)], sorted_indices, sign_x)
                    # return cls._rearrange_org(s_list[-1], sorted_indices, sign_x)

        @staticmethod
        def _rearrange_org(ytilde: np.ndarray, sorted_indices, sign: np.ndarray) -> np.ndarray:
            return ytilde[sorted_indices] * sign

        @classmethod
        def _loop_search_s(cls, y: np.ndarray, nu: np.ndarray) -> list[np.ndarray]:
            # Search s
            y_l1_array = y.cumsum()

            s_arange = np.arange(1, len(y_l1_array) + 1)
            cs_array = cls._calculate_cs(y_l1_array, nu, s_arange)  # y -> y_s
            y_plus = np.append(y, -1)[1:]

            condition4 = (((3 * nu ** (2 / 3)) / (2 ** (4 / 3)))) * s_arange ** (2 / 3)

            # res = (y_l1_array >= condition4) & (y_l1_array >= cls._calculate_condition3(v=nu, q=1 / 2, s=s_arange)) & (
            #         y > cs_array) & (cs_array >= y_plus)
            res = (y_l1_array >= condition4) & (
                    y > cs_array) & (cs_array >= y_plus)
            return [np.maximum(y - cs_array[ind], 0) for ind in np.where(res)[0]]

        @staticmethod
        def _calculate_cs(y_l1_array: np.ndarray, nu: np.ndarray, s) -> np.ndarray:
            return (np.sqrt(3.0) * nu) / (
                    4 * np.sqrt(y_l1_array) * np.cos(
                (1 / 3) * np.arccos(-3 * np.sqrt(3.0) * nu * s * np.power(y_l1_array, -3 / 2) / 4)))

        # @staticmethod
        # def _calculate_condition3(v, q, s):
        #     return (2 - q) * (v * q * s / ((q - 1) ** (1 - q))) ** (1 / (2 - q))
        @staticmethod
        def _calculate_condition3(v, q, s):  # cs
            return (2 - q) * (v * q * s / ((q - 1) ** (1 - q))) ** (1 / (2 - q))


class ProxL1_2over3(GroupProximalOperator):
    """
    Standard implementation of the proximal operator for the \ell_{1,\frac{2}{3}}.
    Ralated Article:
    [1] Lin R, Chen S, Feng H, et al. Computing the Proximal Operator of the $\ell_ {1, q} $-norm for Group Sparsity[J]. arXiv preprint arXiv:2409.14156, 2024.

    """

    def __init__(self, n: int, gLen: int, num_samples: int = 1):
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)
        self.num_samples: int = num_samples

    @classmethod
    def obj(cls, x: np.ndarray, xtilde: np.ndarray, nu: T) -> np.ndarray:
        return super().obj(x, xtilde, nu)

    def prox(self, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        x_subvectors = self._split_vectors(x)
        return np.concatenate([self.L1_2over3_part.prox(x_sub, v) for x_sub in x_subvectors]).reshape(self.num_samples,
                                                                                                      -1)

    @classmethod
    def name(cls):
        return 'group_l1_2over3'

    def _split_vectors(self, x: np.ndarray) -> list[np.ndarray]:
        return np.split(x.flatten(), self.num_subvectors * self.num_samples)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError('You dont need to call this method')

    class L1_2over3_part():

        @staticmethod
        def obj(y: np.ndarray, ytilde: np.ndarray, nu: T) -> np.ndarray:
            return nu * np.power(ytilde.sum(), 2 / 3) + 0.5 * (np.linalg.norm(y - ytilde, 2) ** 2)

        @classmethod
        def prox(cls, x: np.ndarray, nu: T) -> np.ndarray:
            x_abs = np.abs(x)
            condition_1 = 4 * ((2 / 9) ** 0.75) * (nu ** 0.75)
            ys_l1 = x_abs.sum()

            if ys_l1 <= condition_1:  # case1
                return np.zeros_like(x)
            else:
                condition_2 = 2 * ((2 * nu / 3) ** 0.75)
                sign_x = np.sign(x)
                indices = np.argsort(-x_abs)
                y = x_abs[indices]
                if y[-1] > condition_2:
                    return (x_abs - cls._calculate_cs(ys_l1, nu, len(x))) * sign_x
                else:
                    # Search s
                    sorted_indices = np.argsort(indices)
                    s_list = [np.zeros_like(y)]
                    s_list.extend(cls._loop_search_s(y, nu))
                    Js_list = [cls.obj(y, per_ytilde, nu) for per_ytilde in s_list]
                    return cls._rearrange_org(s_list[np.argmin(Js_list)], sorted_indices, sign_x)

        @staticmethod
        def _rearrange_org(ytilde: np.ndarray, sorted_indices, sign: np.ndarray) -> np.ndarray:
            return ytilde[sorted_indices] * sign

        @classmethod
        def _loop_search_s(cls, y: np.ndarray, nu: np.ndarray) -> list[np.ndarray]:
            # Search s
            y_l1_array = y.cumsum()

            s_arange = np.arange(1, len(y_l1_array) + 1)
            cs_array = cls._calculate_cs(y_l1_array, nu, s_arange)
            y_plus = np.append(y, -1)[1:]
            if y.sum() < 1e-5:
                return [np.maximum(y - cs_array[ind], 0) for ind in np.where(y > cs_array)[0]]
            condition4 = ((4 / 3) * nu ** 0.75 * ((2 ** 0.75) / (3 ** 0.5))) * s_arange ** 0.75
            res = (y_l1_array >= condition4) & (
                    y > cs_array) & (
                          cs_array >= y_plus)  # & (y_l1_array >= cls._calculate_condition3(v=nu, q=2 / 3, s=s_arange))
            return [np.maximum(y - cs_array[ind], 0) for ind in np.where(res)[0]]

        @staticmethod
        def _calculate_condition3(v, q, s):
            return (2 - q) * (v * q * s / ((q - 1) ** (1 - q))) ** (1 / (2 - q))

        @staticmethod
        def _calculate_cs(y_l1_array: np.ndarray, nu: np.ndarray, s) -> np.ndarray:
            tmp = y_l1_array ** 4 / 256 - 8 * (nu ** 3) * (s ** 3) / 729
            part_alpha = np.where(tmp < 0, np.sqrt(np.sqrt(8 * (nu ** 3) * (s ** 3) / 729)), np.sqrt(tmp))

            alpha_s = (y_l1_array ** 2 / 16 + part_alpha) ** (1 / 3) + (np.abs(y_l1_array ** 2 / 16 - part_alpha)) ** (
                    1 / 3)
            sqrt_2alpha = np.sqrt(2 * alpha_s)
            return (4.0 * nu / 3.0) / (sqrt_2alpha + np.sqrt(np.abs(2 * y_l1_array / sqrt_2alpha - 2 * alpha_s)))


class ProxL1_1over2_FixParams(ProxL1_1over2):
    """
    Calculate Proximal L1,1/2 (p=1,q=1/2) with fixing params.
     (When the parameters remain unchanged, it is implemented to accelerate the calculation)
    """

    def __init__(self, n: int, gLen: int, num_samples: int = 1, nu: T = None):
        super(ProxL1_1over2_FixParams, self).__init__(n, gLen, num_samples)
        assert nu is not None, "please input nu"
        self.part = self.L1_1over2_part(gLen, nu)

    def prox(self, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        x_subvectors = self._split_vectors(x)
        return np.concatenate([self.part.prox(x_sub, v) for x_sub in x_subvectors]).reshape(self.num_samples, -1)

    class L1_1over2_part(ProxL1_1over2.L1_1over2_part):  # 不继承可提速0.2s
        def __init__(self, gLen, nu):
            self.condition1 = 3 * nu ** (2 / 3) / (2 ** (4 / 3))
            self.condition2 = 1.5 * nu ** (2 / 3)
            self.s_arange = np.arange(1, gLen + 1)
            self.condition3 = self._calculate_condition3(v=nu, q=1 / 2, s=self.s_arange)
            self.condition4 = (3 * nu ** (2 / 3) * (0.5 ** (4 / 3))) * self.s_arange ** (2 / 3)

        def prox(self, x: np.ndarray, nu: T) -> np.ndarray:
            x_abs = np.abs(x)
            if np.linalg.norm(x_abs) < self.condition2:
                return np.zeros_like(x)
            else:
                sign_x = np.sign(x)
                indices = np.argsort(-x_abs)
                y = x_abs[indices]
                if y[-1] > self.condition2:  # case2
                    return (x_abs - self._calculate_cs(x_abs.sum(), nu, len(x))) * sign_x
                else:
                    # Search s
                    sorted_indices = np.argsort(indices)
                    y_tilde, s_max_index = self._loop_search_s2(y, nu, sorted_indices)
                    if y[0] > self.condition2 or s_max_index != -1:  # case 3
                        return y_tilde * sign_x
                    return np.zeros_like(x)

        def _loop_search_s2(self, y: np.ndarray, nu: np.ndarray, sorted_indices):  # -> list[np.ndarray]:
            # Search s
            y_l1_array = y.cumsum()
            cs_array = self._calculate_cs(y_l1_array, nu, self.s_arange)
            y_plus = np.append(y, -1)[1:]

            res = (y_l1_array >= self.condition4) & (y_l1_array >= self.condition3) & (
                    y > cs_array) & (cs_array >= y_plus)
            valid_indices = np.where(res)[0]
            max_index = valid_indices.max() if len(valid_indices) > 0 else -1
            y_tilde_case3 = np.maximum(y[sorted_indices] - cs_array[max_index], 0)
            return y_tilde_case3, max_index


class ProxL1_2over3_FixParams(ProxL1_2over3):
    """
     Calculate Proximal L1,2/3 (p=1,q=2/3) with fixing params.
      (When the parameters remain unchanged, it is implemented to accelerate the calculation)
     """

    def __init__(self, n: int, gLen: int, num_samples: int = 1, nu: T = None):
        super(ProxL1_2over3_FixParams, self).__init__(n, gLen, num_samples)
        assert nu is not None, "please input nu"
        self.part = self.L1_2over3_part(gLen, nu)

    #def prox(self, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        # x_subvectors = self._split_vectors(x)
        # a = np.apply_along_axis(self.part.prox, 1, x_subvectors, v).reshape(self.num_samples, -1)
        # # b=np.concatenate([part.prox(x_sub, v) for index, x_sub in enumerate(x_subvectors)]).reshape(
        # #     self.num_samples, -1)
        #return a
    def prox(self, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        # 将输入重塑为 (样本数, 组数, 每组长度)
        x_reshaped = x.reshape(self.num_samples, -1, self.gLen)
        x_abs = np.abs(x_reshaped)
        sign_x = np.sign(x_reshaped)

        # 向量化计算范数
        norms = np.linalg.norm(x_reshaped, axis=2)
        mask_case1 = norms < self.condition2  # 情况1的掩码

        # 初始化结果矩阵
        non_zero_mask = ~mask_case1
        result = np.zeros_like(x_reshaped)
        return result

    class L1_2over3_part(ProxL1_2over3.L1_2over3_part):

        def __init__(self, gLen, nu):
            self.condition1 = 4 * ((2 / 9) ** 0.75) * (nu ** 0.75)
            self.condition2 = 2 * ((2 * nu / 3) ** 0.75)
            self.s_arange = np.arange(1, gLen + 1)
            self.zeros = np.zeros_like(self.s_arange)
            self.condition3 = self._calculate_condition3(v=nu, q=2 / 3, s=self.s_arange)
            self.condition4 = ((4 / 3) * nu ** 0.75 * (2 ** 0.75 / 3 ** 0.5)) * self.s_arange ** 0.75

        def prox(self, x: np.ndarray, nu: T) -> np.ndarray:
            if np.linalg.norm(x) < self.condition2:
                return self.zeros
            else:
                x_abs = np.abs(x)
                sign_x = np.sign(x)
                indices = np.argsort(-x_abs)
                y = x_abs[indices]
                y_l1_array = y.cumsum()
                if y[-1] > self.condition2:  # case2
                    return (x_abs - self._calculate_cs(y_l1_array[-1], nu, len(x))) * sign_x
                else:
                    sorted_indices = np.argsort(indices)
                    return self._calculate_case34(y, y_l1_array, nu, sorted_indices, sign_x)

        def _calculate_case34(self, y: np.ndarray, y_l1_array, nu: np.ndarray, sorted_indices,
                              sign_x):
            cs_array = self._calculate_cs(y_l1_array, nu, self.s_arange)
            mask = (y_l1_array >= self.condition4) & (y_l1_array >= self.condition3) & (
                    y > cs_array) & (cs_array >= np.append(y, -1)[1:])
            if not np.any(mask):
                return self.zeros
            return np.maximum(y[sorted_indices] - cs_array[np.flatnonzero(mask)[-1]], 0) * sign_x

    @classmethod
    def name(cls):
        return 'group_l1_2over3'

    @staticmethod
    def _rearrange_org(ytilde: np.ndarray, sorted_indices, sign: np.ndarray) -> np.ndarray:
        return ytilde[sorted_indices] * sign
