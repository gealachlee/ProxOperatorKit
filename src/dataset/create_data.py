# ** coding: utf-8 **
"""
@Author: Zhihong Li, Rongrong Lin
@date: 2024/11/25
@description: Create data for Group Sparse.
@version: 2.0
"""

from functools import lru_cache
from numpy.typing import NDArray
from sklearn.preprocessing import normalize
from common.enum import SepSparsityType
import numpy as np

def generate_sensing_mat(m, n):
    A = np.random.normal(size=(m, n))
    return normalize(A, norm='l2', axis=0)

# 生成一个 4x8 的高斯随机矩阵 A
A = np.random.randn(256, 1024)
from  scipy.linalg  import orth
# 使用 scipy.linalg.orth 对矩阵 A 进行正交化
Q = orth(A.T).T #默认是列正交化
# 类型别名
FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


class SensingMatrixGenerator:

    @staticmethod
    @lru_cache(maxsize=128)
    def get_cached_matrix(m: int, n: int) -> FloatArray:
        return generate_sensing_mat(m, n)


class NoiseGenerator:
    @classmethod
    def normal(cls, sig, loc, scale, m):
        return sig * np.random.normal(loc, scale, (m, 1, 1))

    @classmethod
    def laplace(cls, sig, loc, scale, m):
        return sig * np.random.laplace(loc, scale, (m, 1, 1))

    @classmethod
    def rand(cls, sig, m):
        return sig * np.random.rand(m)


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
    m, loc, scale = opts.m, noise_params.loc, noise_params.scale
    if noise_params.dist == 'normal':
        noise = NoiseGenerator.normal(noise_params.sig, loc, scale, m)
    elif noise_params.dist == 'laplace':
        noise = NoiseGenerator.laplace(noise_params.sig, loc, scale, m)
    else:
        noise = NoiseGenerator.rand(noise_params.sig, m)
    print(f'\n-----{noise_params.dist}--{noise_params.sig} noise added-----\n')
    return noise
