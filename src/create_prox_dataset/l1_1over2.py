# 创建prox的输出结果， 用于神经网络训练

# ** coding: utf-8 **
import time

import numpy as np
from munch import DefaultMunch, Munch

from common.enum import DistributionType, SepSparsityType
from config import NoiseParams
from common import JointSparseConfig
from dataset.create_data import create_sc_dataset, SparsityHandler

opts=DefaultMunch(
    K=1000,  # Total Iterations
    objective='Repeat NMSE',
    tau=0.1,  # 0.5  # 'Parameter for reg. term in the objective function'
    m=256,  # 'Number of rows in matrix A'
    n=16,  # 'Number of cols in matrix A'
    data_size=80000,  # 'Number of num_samples'
    dist='normal',  # 'Distribution of entries in the matrix A'
    gLen=16,  # 16  # 'Length of each group
    data_seed=None,  # 'Seed for generating data'
    logger=None,
    sparsity=9,
    plot_figs=True,
    noise_params=NoiseParams(),
    joint_sparse_config=JointSparseConfig(is_joint_sparse=True, mode=SepSparsityType.PERCENTAGE, p=0.2),
)
opts.sep_rate = int(10 * opts.joint_sparse_config.p)
opts.save_dir = f'./'

def probabilistic_zero_out(arr, p=0.3):
    """
    每个元素独立以概率 `p` 置零（可能每行置零数量不同）
    """
    mask = np.random.binomial(1, 1-p, arr.shape)  # 伯努利采样
    return arr * mask

x_test= np.random.normal(0, 0.5, (opts.data_size, 16,1))
x_test =probabilistic_zero_out(x_test, p=opts.joint_sparse_config.p)
x_test= x_test[:, :, 0]
from prox.group_prox.l1psi_dev import *
from prox.group_prox.prox_cl import *

prox_1_1over2 = ProxL1_1over2(16, opts.gLen, opts.data_size)

result=prox_1_1over2.prox(x_test, 0.1)
#
# result.tofile('y.npy')
# x_test.tofile('x.npy')
np.save('x.npy', x_test)
np.save('y.npy', result)