from typing import Union, Tuple

import numpy as np

from common.config import Settings
from models.block_models.block_fista import FISTA
from models.block_models.block_ista import ISTA
from models.joint_models.joint_fista import JointFISTA
from models.joint_models.joint_ista import JointISTA
from models.joint_models.pgac import PGAC
from models.block_models.block_pgac import GROUPPGAC
from prox import ProximalOperator, GroupProximalOperator

__all__ = ['initialize_model', 'get_model_list', 'BlockModelFactory', 'JointModelFactory','FISTA', 'ISTA', 'JointFISTA', 'JointISTA', 'GROUPPGAC',
           'PGAC']

def initialize_model(model_name: str,
                     prox_func: Union[Tuple[ProximalOperator], ProximalOperator,GroupProximalOperator],
                     A: np.ndarray,
                     opts: Settings):
    assert model_name in get_model_list(), f'unknown model name: {model_name}'

    if isinstance(prox_func, tuple):  # joint model
        return JointModelFactory().create_model(model_name=model_name,
                                                A=A, prox_func1=prox_func[0], prox_func2=prox_func[1], tau=opts.tau)
    else:  # block model
        return BlockModelFactory().create_model(model_name=model_name, A=A, prox_func=prox_func, tau=opts.tau)


def get_model_list():
    return [
        'FISTA',
        'ISTA',
        'JointFISTA',
        'JointISTA',
        'PGAC',
        'GROUPPGAC'
    ]


class BlockModelFactory():

    @staticmethod
    def create_FISTA(A: np.ndarray, tau: float, prox_func: Union[ProximalOperator, GroupProximalOperator]):
        return FISTA(A, tau, prox_func)

    @staticmethod
    def create_ISTA(A: np.ndarray, tau: float, prox_func: Union[ProximalOperator, GroupProximalOperator]):
        return ISTA(A, tau, prox_func)

    @staticmethod
    def create_GROUPPGAC(A: np.ndarray, tau: float, prox_func: Union[ProximalOperator, GroupProximalOperator]):
        return GROUPPGAC(A, tau, prox_func)

    def create_model(self, model_name: str, *args, **kwargs):
        assert model_name in get_model_list()
        return getattr(self, f'create_{model_name}')(*args, **kwargs)


class JointModelFactory():
    @staticmethod
    def create_JointFISTA(A: np.ndarray, tau: float, prox_func1: ProximalOperator, prox_func2: GroupProximalOperator):
        return JointFISTA(A, tau, prox_func1, prox_func2)

    @staticmethod
    def create_JointISTA(A: np.ndarray, tau: float, prox_func1: ProximalOperator, prox_func2: GroupProximalOperator):
        return JointISTA(A, tau, prox_func1, prox_func2)

    @staticmethod
    def create_PGAC(A: np.ndarray, tau: float, prox_func1: ProximalOperator, prox_func2: GroupProximalOperator):
        return PGAC(A, tau, prox_func1, prox_func2)

    def create_model(self, model_name: str, *args, **kwargs):
        assert model_name in get_model_list()
        return getattr(self, f'create_{model_name}')(*args, **kwargs)
