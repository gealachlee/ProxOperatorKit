import numpy as np
from munch import DefaultMunch, Munch
from common.enum import DistributionType, SepSparsityType
from config import NoiseParams
from dataset.create_data import create_sc_dataset
from loss import objective_val
from models import BlockModelFactory, JointModelFactory
from prox.group_prox.general import GeneralProxL1Psi
from report import init_settings
from common import JointSparseConfig
import pandas as pd
from munch import DefaultMunch
from prox.group_prox.prox_cl import *
from prox.sep_prox.prox_cl import *
from config import init_log

success_rate_exp_opts = DefaultMunch(
    K=1000,  # Total Iterations
    objective='SUCCESS_TIMES',
    tau=0.1,  # 0.5  # 'Parameter for reg. term in the objective function'
    m=256,  # 'Number of rows in matrix A'
    n=1024,  # 'Number of cols in matrix A'
    data_size=64,  # 'Number of num_samples'
    dist='normal',  # 'Distribution of entries in the matrix A'
    gLen=16,  # 16  # 'Length of each group
    data_seed=9,  # 'Seed for generating data'
    logger=None,
    plot_figs=True,
    noise_params=NoiseParams(),
    algorithm='GROUPIMTC',
    joint_sparse_config=JointSparseConfig(is_joint_sparse=True, mode=SepSparsityType.PERCENTAGE, p=0.5),
    # dynamic_settings= [Munch.fromDict({'sparsity': i}) for i in range(1, 31)]
)

success_rate_exp_opts.save_dir = f'./exp/hist/'
init_log(success_rate_exp_opts)

def calaculate_rate(opts):
    n = opts.n
    gLen = opts.gLen
    N = int(n / gLen)
    success_array = np.zeros(int(N * N)).reshape(N, N)

    sep_sp_rate = np.arange(0, 1, 1 / N)
    group_sp_rate = np.arange(0, 1, 1 / N)

    index = np.arange(64)

    for i_index, sp_rate in zip(index, sep_sp_rate):
        acc_=0
        for j_index, gsp_rate in zip(index, group_sp_rate):
            p = sp_rate #----------------------p表示非0 概率
            sparsity = gsp_rate * N

            opts.joint_sparse_config.p =p

            opts.sparsity = int(sparsity)

            opts.logger(f'\nUsing sparsity: 非零组数{opts.sparsity}（64），组内非零数{(p)*gLen} （16）, 组间非零占比{opts.sparsity/64:.3f},组内非零占比{p}')
            (x_test, d_test), A, b = create_sc_dataset(opts=opts)
            gamma = 1 / (np.linalg.norm(A, 2) ** 2)
            # prox_func2 = ProxL2_1over2(opts.n,
            #                       opts.gLen)  # ProxL1_1over2_FixParams(opts.n, opts.gLen,opts.data_size,nu=opts.tau*gamma)  # ProxL1_1over2_FixParams(opts.n, opts.gLen, opts.data_size,
            # prox_func1 = ProxL1over2()  #
            # prox_1_tl1 = GeneralProxL1Psi(opts.n, opts.gLen,
            #                               subvec_prox=L1_TransformedL1_SubvecProx(opts.gLen, lamb=opts.tau * gamma,
            #                                                                       fix_param=2, precompute=False),
            #                               num_samples=opts.data_size)
            #prox_func=prox_1_tl1
            prox_func = ProxL1_1over2(opts.n, opts.gLen)  # ProxL1_1over2_FixParams(opts.n, opts.gLen, opts.data_size,
            # Create model
            model = BlockModelFactory().create_model(
                model_name=opts.algorithm, A=A, prox_func=prox_func, tau=opts.tau)
            model(d_test, K=opts.K)
            success_rate = objective_val(model.iter_history[-1], d_test, x_test, objective=opts.objective)
            opts.logger(f'Iteration: {opts.K}, success_rate: {success_rate:.4f}\n\n\n')  # .format(, success_rate))

            success_array[int(i_index), int(j_index)] = success_rate
            if np.equal(success_rate,0):
                print('successrate = 0 break')
                break

    return success_array

success_array = calaculate_rate(success_rate_exp_opts)

# import matplotlib.pyplot as plt
# def plot_hist(success_array):
#     plt.imshow(success_array,cmap='coolwarm',interpolation='nearest')
#     plt.xlim(0, 1)  # 设置X轴的范围
#     plt.ylim(0, 1)
#     plt.xticks([0, 32, 63], [0, 0.5, 1])  # X轴的刻度位置和标签
#     plt.yticks([0, 32, 63], [0,0.5,1])
#     plt.colorbar()
#     # Y轴的刻度位置和标签
#     plt.show()
