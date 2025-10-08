from src.prox.group_prox.general import *

__all__=['L1_SubvecProx', 'L1_MCP_SubvecProx', 'L1_SCAD_SubvecProx', 'L1_Transformed']


class L1_MCP_SubvecProx(L1_SubvecProx):

    def __init__(self, gLen: int, lamb: float, fix_param: float, precompute: bool):
        """
        :param lamb : penalty term
        :param nu: func params
        """
        self.element_prox = ProxMCP(fix_param)
        super().__init__(gLen, lamb, self.element_prox)
        self.s_arange = np.arange(1, gLen + 1)
        self.precompute = precompute
        self.fix_param = fix_param
        # if lamb is not None and precompute is True:
        #     self.con1 = self.calculate_con1(lamb)
        #     self.con2 = self.calculate_con2(lamb)
        #     self.con4 = self.calculate_con4(lamb, self.s_arange)

    def calculate_con1(self, lamb: float) -> float:
        return lamb

    def calculate_con2(self, lamb: float) -> float:
        return lamb

    def calculate_con4(self, lamb: float, s_arange: np.ndarray):
        return np.where(lamb * s_arange < self.fix_param, lamb * s_arange, self.fix_param)

    def obj(self, y, ytilde, lamb) -> np.ndarray:
        y_l1 = ytilde.sum()
        return lamb * np.where(y_l1 < self.fix_param, y_l1 - y_l1 ** 2 / (2 * self.fix_param),
                               self.fix_param / 2) + 0.5 * (np.linalg.norm(y - ytilde, 2) ** 2)


class L1_SCAD_SubvecProx(L1_SubvecProx):
    precompute: bool

    def __init__(self, gLen: int, lamb: float, fix_param1: float, fix_param2: float, precompute: bool):
        """
        :param lamb : penalty term
        :param nu: func params
        """
        self.element_prox = ProxSCAD(fix_param1=fix_param1, fix_param2=fix_param2)
        super().__init__(gLen, lamb, self.element_prox)
        self.fix_param1 = fix_param1
        self.fix_param2 = fix_param2
        self.s_arange = np.arange(1, gLen + 1)
        self.precompute = precompute
        # if lamb is not None and precompute is True:
        #     self.con1 = self.calculate_con1(lamb)
        #     self.con2 = self.calculate_con2(lamb)
        #     self.con4 = self.calculate_con4(lamb, self.s_arange)

    def calculate_con1(self, lamb: float) -> float:
        return lamb * self.fix_param1

    def calculate_con2(self, lamb: float) -> float:
        return lamb * self.fix_param1

    def calculate_con4(self, lamb: float, s_arange: np.ndarray) -> np.ndarray:
        return np.where(lamb * s_arange < self.fix_param2,
                        lamb * self.fix_param1 * s_arange,
                        self.fix_param2 * self.fix_param1)

    def obj(self, y, ytilde, lamb) -> np.ndarray:
        y_l1 = ytilde.sum()
        return lamb * np.where(
            y_l1 < self.fix_param1, self.fix_param1 * y_l1,
            np.where(y_l1 <= self.fix_param2 * self.fix_param1,
                     (2 * self.fix_param2 * self.fix_param1 * y_l1 - y_l1 ** 2 - self.fix_param1 ** 2) / (
                             2 * (self.fix_param2 - 1)),
                     (self.fix_param1 ** 2) * 0.5 * (self.fix_param2 + 1))) + 0.5 * (
                np.linalg.norm(y - ytilde, 2) ** 2)


class L1_Transformed(L1_SubvecProx):
    def __init__(self, gLen: int, lamb: float, fix_param: float, precompute: bool):
        self.element_prox = ProxTransformedl1(fix_param)
        super().__init__(gLen, lamb, self.element_prox)
        self.fix_param = fix_param
        self.s_arange = np.arange(1, gLen + 1)
        self.precompute = precompute
        # if precompute is True:
        #     self.con1 = self.calculate_con1(lamb)
        #     self.con2 = self.calculate_con2(lamb)
        #     self.con4 = self.calculate_con4(lamb, self.s_arange)

    def calculate_con1(self, lamb, *args, **kwargs) -> Union[T, int, float, np.ndarray]:
        fix_param = self.fix_param
        return np.where(lamb <= 0.5 * fix_param ** 2 / (fix_param + 1),
                        lamb * (1 + 1 / fix_param),
                        1.5 * (2 * lamb * (fix_param + 1) * fix_param) ** (1 / 3) - fix_param)

    def calculate_con2(self, lamb, *args, **kwargs) -> Union[T, int, float, np.ndarray]:
        fix_param = self.fix_param
        return np.where(lamb <= 0.5 * fix_param ** 2 / (fix_param + 1),
                        lamb * (1 + 1 / fix_param),
                        np.sqrt(2 * lamb * (fix_param + 1)) - 0.5 * fix_param)

    def calculate_con4(self, lamb: T, s_arange: np.ndarray) -> np.ndarray:
        fix_param = self.fix_param

        return np.where(s_arange <= fix_param ** 2 / (2 * lamb * (1 + fix_param)),
                        s_arange * lamb * (1 + 1 / fix_param),
                        1.5 * (2 * s_arange * lamb * (fix_param + 1) * fix_param) ** (1 / 3) - fix_param)

    def obj(self, y: np.ndarray, ytilde: np.ndarray, lamb: T) -> np.ndarray:
        y_l1 = ytilde.sum()
        return lamb * (self.element_prox.a + 1) * y_l1 / (self.element_prox.a + y_l1) + 0.5 * (
                (y - ytilde) ** 2).sum()  # np.linalg.norm(y - ytilde, 2) ** 2
