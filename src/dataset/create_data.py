# ** coding: utf-8 **
"""
@Author: Zhihong Li, Rongrong Lin
@date: 2024/11/25
@description: Create data for Group Sparse.
@version: 2.0
"""
import numpy as np
from dataset import generat_sensing_mat
from common.enum import SepSparsityType


class SparsityHandler:

    @staticmethod
    def handle_group_sparsity(real_signal, opts):
        # def handle_group_sparsity(n, gLen, sparsity):
        #     gNo1 = np.arange(1, int(n / gLen) + 1)
        #     Bs = np.zeros((n, 1))
        #     ActInd = np.random.choice(gNo1, sparsity, replace=False)
        #     for i in range(sparsity):
        #         Bs[((ActInd[i] - 1) * gLen):(ActInd[i] * gLen)] = np.ones((gLen, 1))
        #     return Bs
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


def create_sc_dataset(opts=None) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    :param opts: options in config.py
    :return:  (x_tensor, d_tensor), A, b
    """
    if opts is None:
        from config import Opts
        opts = Opts
    np.random.seed(opts.data_seed)
    m, n = opts.m, opts.n
    sparsity = opts.sparsity
    # test_size = opts.data_size

    A = generat_sensing_mat(m, n)
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
    m = opts.m
    loc, scale = noise_params.loc, noise_params.scale
    if noise_params.dist == 'normal':
        noise = noise_params.sig * np.random.normal(loc, scale, (m, 1, 1))
    elif noise_params.dist == 'laplace':
        noise = noise_params.sig * np.random.laplace(loc, scale, (m, 1, 1))
    else:
        noise = noise_params.sig * np.random.rand(m)
    print(f'\n-----{noise_params.dist}--{noise_params.sig} noise added-----\n')
    return noise
