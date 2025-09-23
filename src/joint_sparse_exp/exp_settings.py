# ** coding: utf-8 **
import time

from munch import DefaultMunch, Munch

from common.enum import DistributionType, SepSparsityType
from config import NoiseParams
from common import JointSparseConfig

__all__ = ['exp2_opts']
#
# exp2_opts = DefaultMunch(
#     K=1000,  # Total Iterations
#     objective='Repeat NMSE',
#     tau=0.1,  # 0.5  # 'Parameter for reg. term in the objective function'
#     m=256,  # 'Number of rows in matrix A'
#     n=1024,  # 'Number of cols in matrix A'
#     data_size=64,  # 'Number of num_samples'
#     # Generate Data Settings
#     dist='normal',  # 'Distribution of entries in the matrix A'
#     sparsity=8,  # 8  # 'Sparsity' # 非零组数
#     gLen=16,  # 16  # 'Length of each group
#     data_seed=10,  # 'Seed for generating data'
#     logger=None,
#     plot_figs=True,
#     noise_params=NoiseParams(),
#     joint_sparse_config=JointSparseConfig(is_joint_sparse=True, mode=SepSparsityType.PERCENTAGE, p=0.5), # p指的是非零概率
#
# )
exp2_opts=DefaultMunch(
    K=1000,  # Total Iterations
    objective='Repeat NMSE',
    tau=0.05,  # 0.5  # 'Parameter for reg. term in the objective function'
    m=256,  # 'Number of rows in matrix A'
    n=1024,  # 'Number of cols in matrix A'
    data_size=1,  # 'Number of num_samples'
    dist='normal',  # 'Distribution of entries in the matrix A'
    gLen=16,  # 16  # 'Length of each group
    data_seed=1,  # 'Seed for generating data'
    logger=None,
    sparsity=9,
    plot_figs=True,
    noise_params=NoiseParams(),
    joint_sparse_config=JointSparseConfig(is_joint_sparse=True, mode=SepSparsityType.PERCENTAGE, p=0.5),
)

exp2_opts.sep_rate = int(10 * exp2_opts.joint_sparse_config.p)
exp2_opts.save_dir = f'./exp/{exp2_opts.joint_sparse_config.mode.value}/'