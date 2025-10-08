from prox import GroupProximalOperator
from prox.sep_prox.prox_cl import *


class ProxL2_0(GroupProximalOperator):
    latex_name = r'$L_{2,|·|_0}$'

    def __init__(self, n: int, gLen: int):
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)

    @classmethod
    def obj(cls, x: np.ndarray, xtilde: np.ndarray, nu: float) -> np.ndarray:
        return super().obj(x, xtilde, nu)

    def prox(self, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        x_subvectors = self._split_vectors(x)
        norms = np.stack([np.linalg.norm(subvec, ord=2, axis=1) for subvec in x_subvectors], axis=0)
        tao = np.where(norms <= np.sqrt(2 * v), 0.0, 1)
        x_subvectors = [tao[i].reshape(-1, 1) * x_subvectors[i] for i in range(self.num_subvectors)]
        return self._concentrate_vectors(x_subvectors)

    @classmethod
    def name(cls) -> str:
        return 'group_l2_0'

    def _split_vectors(self, x: np.ndarray) -> list[np.ndarray]:
        return np.split(x, self.num_subvectors, axis=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(x_subvectors, axis=1)


# Group proximal operators
class ProxL2_1(GroupProximalOperator):
    latex_name = r'$L_{2,|\cdot|}$'

    def __init__(self, n: int, gLen: int):
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)

    @classmethod
    def obj(cls, x: np.ndarray, xtilde: np.ndarray, nu: float) -> np.ndarray:
        raise NotImplementedError('Not implemented yet')

    def prox(self, x: np.ndarray, v: float, *args, **kwargs) -> np.ndarray:
        x_subvectors = self._split_vectors(x)
        norms = np.stack([np.linalg.norm(subvec, ord=2, axis=1) for subvec in x_subvectors], axis=0)
        tao = np.where(norms < v, 0, (1 - v / (norms)))
        x_subvectors = [tao[i].reshape(-1, 1) * x_subvectors[i] for i in range(self.num_subvectors)]
        return self._concentrate_vectors(x_subvectors)

    @classmethod
    def name(cls) -> str:
        return 'group_l2_1'

    def _split_vectors(self, x: np.ndarray) -> list[np.ndarray]:
        return np.split(x, self.num_subvectors, axis=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(x_subvectors, axis=1)


class ProxL2_2over3(GroupProximalOperator):
    latex_name = r'$L_{2,|\cdot|^{2/3}}$'

    def __init__(self, n: int, gLen: int):
        self.n = n
        self.gLen = gLen
        self.num_subvectors = self.get_num_subvectors(n, gLen)

    @classmethod
    def name(cls) -> str:
        return 'group_l2_2over3'

    @classmethod
    def obj(cls, x: np.ndarray, xtilde: np.ndarray, nu: float) -> np.ndarray:
        raise NotImplementedError('Not implemented yet')

    def prox(self, x: np.ndarray, v: float, *args, **kwargs) -> np.ndarray:
        x_subvectors = self._split_vectors(x)
        norms = np.stack([np.linalg.norm(subvec, axis=1) for subvec in x_subvectors], axis=0)

        t = (1 / 16 + np.sqrt(1 / 256 - (8 * (v ** 3) / (729 * norms ** 4)))) ** (1 / 3) + (
                1 / 16 - np.sqrt(1 / 256 - (8 * (v ** 3) / (729 * norms ** 4)))) ** (1 / 3)
        tao = np.where(norms <= 2 * ((2 / 3) * v ** 0.75), 0.0,
                       (1.0 / 8.0) * (np.sqrt(2.0 * t) + (
                           np.sqrt(2.0 / np.sqrt(2.0 * t) - 2.0 * t))) ** 3)
        x_subvectors = [tao[i].reshape(-1, 1) * x_subvectors[i] for i in range(self.num_subvectors)]

        return self._concentrate_vectors(x_subvectors)

    def _split_vectors(self, x: np.ndarray) -> list[np.ndarray]:
        return np.split(x, self.num_subvectors, axis=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(x_subvectors, axis=1)


class ProxL2_1over2(GroupProximalOperator):
    latex_name = r'$L_{2,|\cdot|^{1/2}}$'

    def __init__(self, n: int, gLen: int):
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)

    @classmethod
    def obj(cls, x: np.ndarray, xtilde: np.ndarray, nu: float) -> np.ndarray:
        raise NotImplementedError('Not implemented yet')

    def prox(self, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        x_subvectors = self._split_vectors(x)

        norms = np.stack([np.linalg.norm(subvec, ord=2, axis=1) for subvec in x_subvectors], axis=0)

        break_point = 1.5 * v ** (2 / 3)
        tao = np.where(norms <= break_point, 0.0, (2 / 3) * (1.0 + np.cos((2 / 3) *
                                                                          (np.arccos(
                                                                              -3 ** 1.5 / 4 * v * (norms ** -1.5)
                                                                          )))))
        x_subvectors = [tao[i].reshape(-1, 1) * x_subvectors[i] for i in range(self.num_subvectors)]

        return self._concentrate_vectors(x_subvectors)

    @classmethod
    def name(cls):
        return 'group_l2_1over2'

    def _split_vectors(self, x: np.ndarray) -> list[np.ndarray]:
        return np.split(x, self.num_subvectors, axis=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(x_subvectors, axis=1)


class ProxL2_Arctan(GroupProximalOperator):
    latex_name = r'$L_{2,{\rm Arctan}}$'

    def __init__(self, n: int, gLen: int, c: float):
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)
        self.ele_proxfunc = ProxArctan(c=c)
        self.c = c

    def prox(self, x: np.ndarray, v, *args, **kwargs) -> np.ndarray:
        x_subvectors = self._split_vectors(x)

        norms = np.stack([np.linalg.norm(subvec, ord=2, axis=1) for subvec in x_subvectors], axis=0)

        break_point = self.c * v / 2
        tao = np.where(norms <= break_point, 0.0, self.ele_proxfunc.prox(norms, v) / norms)
        x_subvectors = [tao[i].reshape(-1, 1) * x_subvectors[i] for i in range(self.num_subvectors)]

        return self._concentrate_vectors(x_subvectors)

    @classmethod
    def name(cls):
        return 'group_l2_arctan'

    def _split_vectors(self, x: np.ndarray) -> list[np.ndarray]:
        return np.split(x, self.num_subvectors, axis=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(x_subvectors, axis=1)


class ProxL2_CappedL1over2(GroupProximalOperator):
    latex_name: str = r'$L_{2,{\rm CL1/2}}$'

    def __init__(self, n: int, gLen: int, gamma: float):
        super().__init__()
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)
        self.gamma = gamma
        assert self.gamma > 0, ValueError('gamma must be greater than 0')

    @classmethod
    def prox_l1over2(cls, x, threshold):  # succeed
        condition = 1.5 * threshold ** (2.0 / 3.0)
        return np.where(np.abs(x) <= condition, 0.0,
                        (4.0 / 3.0) * x * (np.cos((1.0 / 3.0) * np.arccos(
                            -3 ** 1.5 / 4 * threshold * (np.abs(x) ** -1.5)
                        )) ** 2)
                        )

    @classmethod
    def obj(cls, x: np.ndarray, xtilde: np.ndarray, v: np.ndarray, gamma) -> np.ndarray:
        f = (x / gamma) ** 0.5
        return 0.5 * (x - xtilde) ** 2 + v * np.where(f <= 1, f, 1)

    def prox(self, x: np.ndarray, v: np.ndarray, *args, **kwargs) -> np.ndarray:
        x_subvectors = self._split_vectors(x)

        norms = np.stack([np.linalg.norm(subvec, ord=2, axis=1) for subvec in x_subvectors], axis=0)

        gamma = self.gamma
        con = self.prox_l1over2(norms, v / (gamma) ** 0.5)
        u1_star = np.where(con <= gamma, con, gamma)
        u2_star = np.where(norms < gamma, gamma, norms)

        condition = self.obj(u1_star, norms, v=v, gamma=self.gamma) <= self.obj(u2_star, norms, v=v,
                                                                                gamma=self.gamma)
        tao = np.where(condition,
                       u1_star / (norms),
                       u2_star / (norms))

        x_subvectors = [tao[i].reshape(-1, 1) * x_subvectors[i] for i in range(self.num_subvectors)]

        return self._concentrate_vectors(x_subvectors)

    @classmethod
    def name(cls):
        return 'ProxCappedL1over2'

    def _split_vectors(self, x: np.ndarray) -> list[np.ndarray]:
        return np.split(x, self.num_subvectors, axis=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(x_subvectors, axis=1)


class ProxL2_LogSum(GroupProximalOperator):
    latex_name = r'$L_{2,{\rm LOG}}$'

    def __init__(self, n: int, gLen: int, epsilon):
        super().__init__()
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)
        self.epsilon = epsilon

    def prox(self, x: np.ndarray, v: np.ndarray, *args, **kwargs) -> np.ndarray:
        x_subvectors = self._split_vectors(x)
        norms = np.stack([np.linalg.norm(subvec, ord=2, axis=1) for subvec in x_subvectors], axis=0)
        if np.sqrt(v) < self.epsilon:
            con = np.logical_or(
                ((norms + self.epsilon) ** 2 - 4 * v <= 0) * (norms <= v / self.epsilon),
                norms == 0)

        else:
            func = lambda t: (1 / (2 * v)) * (self.rx(t, v, self.epsilon) - t) ** 2 + np.log(
                1 + self.rx(t, v, self.epsilon) / self.epsilon) - (1 / (2 * v)) * t ** 2
            b = self.binary_search(func=func, target=0, low=2 * np.sqrt(v) - self.epsilon, high=v / self.epsilon)
            con = norms <= b
        tao = np.where(
            con, 0, self.rx(norms, v, self.epsilon) / norms)
        x_subvectors = [tao[i].reshape(-1, 1) * x_subvectors[i] for i in range(self.num_subvectors)]
        return self._concentrate_vectors(x_subvectors)

    @classmethod
    def name(cls) -> str:
        return 'ProxL2LogSum'

    def _split_vectors(self, x: np.ndarray) -> list[np.ndarray]:
        return np.split(x, self.num_subvectors, axis=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(x_subvectors, axis=1)

    @staticmethod
    def rx(x, v, epsl):
        return 0.5 * (x - epsl) + 0.5 * np.sqrt((x + epsl) ** 2 - 4 * v)

    @classmethod
    def binary_search(cls, func, target, low, high, tol=0.2):
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
