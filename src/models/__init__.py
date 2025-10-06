from typing import Callable, Union, Tuple

import numpy as np

from common.config import Settings
from models.block_models.block_fista import FISTA
from models.block_models.block_ista import ISTA
from models.joint_models.joint_fista import JointFISTA
from models.joint_models.joint_ista import JointISTA
from models.joint_models.IMTC import IMTC
from models.block_models.IMTC import GROUPIMTC
from prox import ProximalOperator

__all__ = ['initialize_model', 'get_model_list', 'BlockModelFactory', 'JointModelFactory','FISTA', 'ISTA', 'JointFISTA', 'JointISTA', 'IMTC', 'GROUPIMTC']

def initialize_model(model_name: str,
                     prox_func: Union[Tuple[ProximalOperator], ProximalOperator],
                     A: np.ndarray,
                     opts: Settings):
    assert model_name in get_model_list(), f'unknown model name: {model_name}'
    assert isinstance(prox_func, (ProximalOperator, tuple)), f'unknown proximal operator: {prox_func}'

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
        'IMTC',
        'GROUPIMTC'
    ]


class BlockModelFactory():

    @staticmethod
    def create_FISTA(A: np.ndarray, tau: float, prox_func: Callable):
        return FISTA(A, tau, prox_func)

    @staticmethod
    def create_ISTA(A: np.ndarray, tau: float, prox_func: Callable):
        return ISTA(A, tau, prox_func)

    @staticmethod
    def create_GROUPIMTC(A: np.ndarray, tau: float, prox_func: Callable):
        return GROUPIMTC(A, tau, prox_func)

    def create_model(self, model_name: str, *args, **kwargs):
        assert model_name in get_model_list()
        return getattr(self, f'create_{model_name}')(*args, **kwargs)


class JointModelFactory():
    @staticmethod
    def create_JointFISTA(A: np.ndarray, tau: float, prox_func1: Callable, prox_func2: Callable):
        return JointFISTA(A, tau, prox_func1, prox_func2)

    @staticmethod
    def create_JointISTA(A: np.ndarray, tau: float, prox_func1: Callable, prox_func2: Callable):
        return JointISTA(A, tau, prox_func1, prox_func2)

    @staticmethod
    def create_IMTC(A: np.ndarray, tau: float, prox_func1: Callable, prox_func2: Callable):
        return IMTC(A, tau, prox_func1, prox_func2)

    def create_model(self, model_name: str, *args, **kwargs):
        assert model_name in get_model_list()
        return getattr(self, f'create_{model_name}')(*args, **kwargs)
