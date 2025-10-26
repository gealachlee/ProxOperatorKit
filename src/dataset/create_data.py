# ** coding: utf-8 **
"""
@Author: Zhihong Li, Rongrong Lin
@date: 2024/11/25
@description: Create data for Group Sparse.
@version: 2.0
"""

from functools import lru_cache
from typing import Dict, Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import normalize

from common.enum import SepSparsityType

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


def generate_sensing_mat(m, n):
    A = np.random.normal(size=(m, n))
    return normalize(A, norm='l2', axis=0)



class SensingMatrixGenerator:

    @staticmethod
    @lru_cache(maxsize=128)
    def get_cached_matrix(m: int, n: int) -> FloatArray:
        return generate_sensing_mat(m, n)




class NoiseGenerator:
    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        """注册装饰器"""

        def decorator(func: Callable) -> Callable:
            cls._registry[name] = func
            return func

        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> np.ndarray:
        """创建指定类型的噪声"""
        if name not in cls._registry:
            raise ValueError(f"未知的噪声类型: {name}。可用的类型: {list(cls._registry.keys())}")

        generator_func = cls._registry[name]
        return generator_func(*args, **kwargs)

    @classmethod
    def list_generators(cls) -> list:
        """列出所有可用的噪声生成器"""
        return list(cls._registry.keys())

@NoiseGenerator.register("normal")
def normal(sig: float, loc: float = 0.0, scale: float = 1.0, m: int = 1) -> np.ndarray:
    """正态分布噪声"""
    return sig * np.random.normal(loc, scale, (m, 1, 1))


@NoiseGenerator.register("laplace")
def laplace(sig: float, loc: float = 0.0, scale: float = 1.0, m: int = 1) -> np.ndarray:
    """拉普拉斯分布噪声"""
    return sig * np.random.laplace(loc, scale, (m, 1, 1))


@NoiseGenerator.register("uniform")
def uniform(sig: float, m: int = 1) -> np.ndarray:
    """均匀分布噪声"""
    return sig * np.random.rand(m, 1, 1)


@NoiseGenerator.register("gaussian")
def gaussian(sig: float, mean: float = 0.0, std: float = 1.0, m: int = 1) -> np.ndarray:
    """高斯分布噪声"""
    return sig * np.random.normal(mean, std, (m, 1, 1))



class SparsityHandler:

    @staticmethod
    def handle_group_sparsity(real_signal, opts):
        size = opts.data_size
        n = opts.n
        gLen = opts.gLen
        sparsity = opts.sparsity

        Bs_tensor = np.zeros((size, n, 1))
        gNo1 = np.arange(1, int(n / gLen) + 1)

        for t in range(size):
            Bs = np.zeros((n, 1))
            ActInd = np.random.choice(gNo1, sparsity, replace=False)
            for i in range(sparsity):
                Bs[((ActInd[i] - 1) * gLen):(ActInd[i] * gLen)] = np.ones((gLen, 1))
            Bs_tensor[t] = Bs
        return real_signal * Bs_tensor

    @staticmethod
    def handle_sep_sparsity(real_signal, opts):
        size = opts.data_size
        n = opts.n

        join_sparse_cfg = opts.joint_sparse_config
        if join_sparse_cfg.is_joint_sparse is False:
            return real_signal
        if join_sparse_cfg.mode == SepSparsityType.BINOMIAL:
            return real_signal * np.random.binomial(size=[size, n, 1], n=1, p=1 - join_sparse_cfg.p)
        elif join_sparse_cfg.mode == SepSparsityType.PERCENTAGE:
            gLen = opts.gLen
            subvec_list = np.split(real_signal.flatten(), (n // gLen) * size)
            for index, pervec in enumerate(subvec_list):
                indices = np.random.choice(gLen, size=int(opts.gLen * (join_sparse_cfg.p)), replace=False)
                subvec_list[index][indices] = 0
            real_signal = np.array(subvec_list).reshape((size, n, 1))
            return real_signal
        else:
            non_sparse_len = np.where(real_signal)
        return real_signal


def create_sc_dataset(opts) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    :param opts: options in config.py
    :return:  (x_tensor, d_tensor), A, b
    """
    np.random.seed(opts.data_seed)
    m, n = opts.m, opts.n
    sparsity = opts.sparsity
    # test_size = opts.data_size

    A = SensingMatrixGenerator.get_cached_matrix(m, n)
    c = np.random.normal(0, 1, (opts.data_size, n, 1))

    c = SparsityHandler.handle_group_sparsity(c, opts)
    if opts.joint_sparse_config.is_joint_sparse:
        c = SparsityHandler.handle_sep_sparsity(c, opts)

    if opts.noise_params is not None:
        noise = make_noise_data(opts)
    else:
        noise = 0
    b = A.dot(c) + noise

    # # Transform x and d into tensors
    x_tensor = c[:, :, 0]
    d_tensor = b[:, :, 0].T
    return (x_tensor, d_tensor), A, b


def make_noise_data(opts):
    noise_params = opts.noise_params
    m, loc, scale,sig = opts.m, noise_params.loc, noise_params.scale,noise_params.sig
    noise=NoiseGenerator.create(noise_params.dist , sig=sig, loc=loc, scale=scale, m=m)
    print(f'\n-----{noise_params.dist}--{noise_params.sig} noise added-----\n')
    return noise



