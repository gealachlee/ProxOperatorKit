import numpy as np
from collections import namedtuple

from prox import ProximalOperator


class ProxL1(ProximalOperator):
    latex_name = r'$\ell_1$'

    @classmethod
    def prox(cls, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - v, 0)

    @classmethod
    def name(cls):
        return 'L1'

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    # def latex_name(cls):
    #     return r'$\ell_1$'


class ProxL1over2(ProximalOperator):
    latex_name = r'$\ell_{1/2}$'

    @classmethod
    def prox(cls, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        condition = 1.5 * v ** (2.0 / 3.0)
        absx = np.abs(x)
        return np.where(absx <= condition, 0.0,
                        (2 / 3) * x * (1 + np.cos((2 / 3) * np.arccos(
                            (-(3 ** 1.5) / 4) * v * (absx ** -1.5)))))

    @classmethod
    def name(cls):
        return 'L1over2'

    # @classmethod
    # def latex_name(cls):
    #     return r'$\ell_{1/2}$'

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class ProxL2over3(ProximalOperator):
    latex_name = r'$\ell_{2/3}$'
    @classmethod
    def prox(cls, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        x = np.where(
            (x ** 4 / 256 - (8 * (v ** 3) / 729)) <= 0,
            np.sqrt(np.sqrt(8 * 256 * (v ** 3) / 729)),
            x)
        t = (x ** 2 / 16 + np.sqrt(x ** 4 / 256 - (8 * (v ** 3) / 729))) ** (1 / 3) + (
                x ** 2 / 16 - np.sqrt(x ** 4 / 256 - (8 * (v ** 3) / 729))) ** (1 / 3)
        tao = np.where(np.abs(x) <= 2 * ((2 / 3) * v ** 0.75), 0.0,
                       (1 / 8.0) * np.sign(x) * (np.sqrt(2.0 * t) + (
                           np.sqrt(2.0 * np.abs(x) / np.sqrt(2.0 * t) - 2 * t))) ** 3)
        return tao

    @classmethod
    def name(cls):
        return 'L2over3'

    # @classmethod
    # def latex_name(cls):
    #     return r'$\ell_{2/3}$'

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class ProxL0(ProximalOperator):
    latex_name = r'$\ell_{0}$'

    @classmethod
    def prox(cls, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        condition = np.sqrt(2 * v * 1.0)
        absx = np.abs(x)
        return np.where(absx <= condition, 0.0, x)

    @classmethod
    def name(cls):
        return 'L0'

    # @classmethod
    # def latex_name(cls):
    #     return r'$\ell_0$'

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class ProxArctan(ProximalOperator):
    latex_name = r'${{\rm {Arctan}}$'
    """
    Related paper:
        [1] He Z, Shu Q, Wen J, et al. A Novel Iterative Thresholding Algorithm for Arctangent
    Regularization Problem[C]//ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal
    Processing (ICASSP). IEEE, 2024: 9651-9655.
    """

    def __init__(self, c: float):
        self.c = c
        assert c > 0, ValueError("c must be greater than 0")

    def prox(self, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        c = self.c
        p_absx = 1 / (3 * c ** 2) - (x ** 2) / 9
        q_absx = v / (4 * c) - np.abs(x) / (3 * c ** 2) - np.abs(x) ** 3 / 27
        assert v < 16 * np.sqrt(3.0) / (9 * c ** 2), ValueError("v is too large")
        r = np.sign(q_absx) * np.sqrt(np.abs(p_absx))
        nv = np.where(p_absx < 0, np.arccosh(q_absx / (r ** 3)), np.arcsinh(q_absx / (r ** 3)))
        tau = np.where(
            np.logical_and(p_absx < 0, np.abs(x) > v * c / 2), -2 * r * np.cosh(nv / 3) + np.abs(x) / 3,
            np.where(
                np.logical_and(p_absx > 0, np.abs(x) > v * c / 2),
                -2 * r * np.sinh(nv / 3) + np.abs(x) / 3, 0
            )
        )
        return np.where(np.abs(x) > v * c / 2, np.sign(x) * tau, 0)

    @classmethod
    def name(cls):
        return "arctan"

    # @classmethod
    # def latex_name(cls):
    #     return r'${\rm{Arctan}}$'

    @classmethod
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class ProxMCP(ProximalOperator):
    latex_name = r'${\rm MCP}$'
    def __init__(self, fix_params):
        self.fix_params = fix_params
        assert fix_params > 1

    def prox(self, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        absx = np.abs(x)
        return np.where(absx <= v, 0,
                        np.where(absx < self.fix_params,
                                 (self.fix_params / (self.fix_params - v)) * (absx - v) * np.sign(x), x))

    @classmethod
    def name(cls):
        return 'MCP'

    def calculate_subderivative(self, u, *args, **kwargs):
        """ for u>0
        $\psi'(t)=\max\{0,1-\frac{t}{\nu}\}$
        :param *args:
        :param **kwargs:
        :param u:
        :return:
        """
        return np.where(1 - (u / self.fix_params) > 0, 1 - (u / self.fix_params), 0)


class ProxSCAD(ProximalOperator):
    latex_name =r'${\rm {SCAD}}$'
    def __init__(self, fix_param1, fix_param2):
        self.fix_param1 = fix_param1
        self.fix_param2 = fix_param2

        assert fix_param1 > 0
        assert fix_param2 > 2

    def prox(self, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        lamb = v * self.fix_param1

        absx = np.abs(x)

        # return np.where(absx < 2 * lamb, 0,
        #                 np.maximum(x - lamb * np.sign(x),
        #                            np.where(absx <= self.fix_param2 * lamb,
        #                                     ((self.fix_param2 - 1) * absx - lamb * self.fix_param2) * np.sign(
        #                                         x) / (self.fix_param2 - 2),
        #                                     x)
        #                            ))

        return np.where(absx <= 2 * lamb,
                        np.maximum(np.sign(x) * (absx - lamb), 0),
                        np.where(absx <= self.fix_param2 * lamb,
                                 ((self.fix_param2 - 1) * absx - lamb * self.fix_param2) * np.sign(x) / (
                                         self.fix_param2 - 2), x))
        # return np.where(absx <= v * self.fix_param1,0,
        #                 np.where(
        #                     absx<=self.fix_param1*(v+1),
        #                          np.sign(x) * (absx - self.fix_param1*v),
        #                         np.where(
        #                             absx <= self.fix_param2 *self.fix_param1,
        #                                  ((self.fix_param2 - 1) * absx - self.fix_param1*v * self.fix_param2) * np.sign(x)
        #                                  / (
        #                                  self.fix_param2 - 1-v), x)
        #                 )
        #                 )

    @classmethod
    def name(cls):
        return 'SCAD'

    # @classmethod
    # def latex_name(cls):
    #     return r'$\rm{SCAD}$'

    def calculate_subderivative(self, u, *args, **kwargs):
        """ for u>0

        :param *args:
        :param **kwargs:
        :param u:
        :return:
        """
        return np.where(u <= self.fix_param1, self.fix_param1,
                        np.where(u < self.fix_param2 * self.fix_param1,
                                 (self.fix_param2 * self.fix_param1 - u) / (self.fix_param2 - 1), 0))


class ProxTransformedl1(ProximalOperator):

    latex_name= r'${\rm{TL1}}$'
    def __init__(self, fix_param1):
        self.a = fix_param1

    def prox(self, x, v, *args, **kwargs):
        a = self.a
        absx = np.abs(x)
        t = np.where(v <= a ** 2 / (2 * (a + 1)), v * (a + 1) / a,
                     np.sqrt(2 * v * (a + 1)) - a / 2)

        l = (1 - 27 * v * a * (a + 1) / (2 * (a + absx) ** 3))
        tao = np.where(
            np.logical_or(absx <= t, np.abs(l) > 1), 0,
            np.sign(x) * ((2 / 3) * (a + absx) *
                          np.cos(
                              np.arccos(
                                  l
                              ) / 3
                          ) - 2 * a / 3 + absx / 3))
        return tao

    @classmethod
    def name(cls):
        return 'Tl1'

    def calculate_subderivative(self, u: np.ndarray, *args, **kwargs) -> np.ndarray:
        """ u>0
        :param u:
        :param args:
        :param kwargs:
        :return:
        """
        a = self.a
        return (a + 1) * a / ((u + a) ** 2)

    # @classmethod
    # def


class ProxCappedL1(ProximalOperator):
    latex_name = r'$\rm{CL1}$'
    def __init__(self, fix_param):
        self.fix_param = fix_param
        assert fix_param > 0

    def prox(self, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        if v / (self.fix_param ** 2) < 2:
            absx = np.abs(x)
            return np.where(absx <= self.fix_param + v / (2 * self.fix_param),
                            np.maximum(absx - v / self.fix_param, 0) * np.sign(x), x)
        else:
            return proxl0(x, lamb=v)

    @classmethod
    def name(cls):
        return 'CappedL1'




def proxl0(x, lamb) -> np.ndarray:
    return np.where(np.abs(x) <= np.sqrt(2 * lamb), 0, x)


class ProxCapped1over2(ProximalOperator):
    latex_name = r'$\rm{CL1/2}$'
    p: float = 0.5

    def __init__(self, fixparam: float):

        self.PreComputePoints = namedtuple('PreComputePoints',
                                           ['taup', 'sqrt2lamb', 'lamb_over_v2', 'c', 'C_lamb_v_p', 'lamb_over_phalf'])

        self.fixparam = fixparam
        assert fixparam > 0

    def prox(self, x: np.ndarray, lamb) -> np.ndarray:

        """
        使用闭式解的方法计算prox_{λ/v^p CL1/2}(x)
        :param x:
        :return:
        """
        v = self.fixparam

        c = 1.5 * (lamb ** (2 / 3)) / (v ** (1 / 3))
        self.precomput_points = self.PreComputePoints(
            c=c,
            taup=self.calculate_taup(),
            sqrt2lamb=self.calculate_sqrt2lamb(lamb),  # sqrt(2λ)
            lamb_over_v2=lamb / (v ** 2),
            lamb_over_phalf=lamb / (v ** self.p),  # λ/v^p
            C_lamb_v_p=self.calculate_C_lamb_v_p(lamb, v, c)  # 用二分法找的C_{λ,v,p}点
        )
        if self.precomput_points.lamb_over_v2 >= self.precomput_points.taup:
            print('calculate_l0')
            return proxl0(x, lamb)
        else:
            absx = np.abs(x)
            return np.where(absx > self.precomput_points.C_lamb_v_p,
                            x,
                            ProxL1over2.prox(x, v=self.precomput_points.lamb_over_phalf))

    def calculate_taup(self):
        return 512 / 729  # ((2 ** (4 - 3 * p)) * (1 - p) ** (2 - 2 * p) / ((2 - p) ** (4 - 2 * p))) ** (1 / p)

    def calculate_sqrt2lamb(self, lamb):
        return np.sqrt(2 * lamb)

    def calculate_C_lamb_v_p(self, lamb, v, c):
        p = self.p

        func = lambda y: lamb * (ProxL1over2.prox(y, lamb / (v ** p)) / v) ** p + 0.5 * (
                ProxL1over2.prox(y, lamb / (v ** p)) - y) ** 2 - lamb
        return self.binary_search(func, 0.0, c, v + lamb / (2 * v))

    @classmethod
    def binary_search(cls, func, target, low, high, tol=1e-7):
        """
        使用二分法查找函数 func 的根，使得 func(x) = target。

        :param func: 目标函数
        :param target: 目标值
        :param low: 搜索区间下限
        :param high: 搜索区间上限
        :param tol: 容差
        :return: 近似根
        """

        while low <= high:
            mid = (low + high) / 2
            mid_value = func(mid)

            if np.abs(mid_value - target) < tol:
                return mid
            elif mid_value < target:
                low = mid
            else:
                high = mid

    @classmethod
    def name(cls):
        return 'CappedL1over2'



