import numpy as np

from models.base import JointModel

from prox import ProximalOperator, GroupProximalOperator


class JointISTA(JointModel):
    def __init__(self, A: np.ndarray, tau,
                 prox_func1: ProximalOperator,
                 prox_func2: GroupProximalOperator):
        super().__init__(A, tau, prox_func1, prox_func2)
        self.tau = tau
        self.prox_func1 = prox_func1
        self.prox_func2 = prox_func2
        self.m, self.n = self.A.shape
        self.L_np = np.linalg.norm(np.matmul(A.transpose(), A), ord=2)
        self.gamma1 = 0.05 * (1 / np.linalg.norm(A, 2) ** 2)
        self.gamma2 = 0.95 * (1 - self.gamma1)

    def name(self):
        return f'ISTA with joint sparse {self.prox_func1.name()} and {self.prox_func2.name()}'

    def T(self, x, d, **kwargs):
        tau = kwargs.get('tau', self.tau)
        index = kwargs.get('index', -1)
        assert index >= 0

        r = (self.A @ x.T).T - d
        z = x - self.gamma1 * 2 * (self.A.T @ r.T).T

        Tx = self.prox_func2(self.prox_func1(z, self.gamma1 * tau), self.gamma1 * tau)
        return Tx
