import numpy as np
from munch import DefaultMunch, Munch
from common.enum import DistributionType, SepSparsityType
from config import NoiseParams
from prox.group_prox.general import GeneralProxL2Psi
from report import init_settings
from common import JointSparseConfig
import pandas as pd
from munch import DefaultMunch
from config import init_log
from dataset.create_data import create_sc_dataset
from loss import objective_val
from models import JointModelFactory, BlockModelFactory
from prox.group_prox import *
from prox.sep_prox.prox_cl import *
from prox.group_prox.l2psi.prox_cl import *
from joint_sparse_exp.exp_settings import exp2_opts
from report import init_settings
from utils import save_df

success_rate_exp_opts = DefaultMunch(
    K=1000,  # Total Iterations
    objective='SUCCESS_TIMES',
    tau=0.05,  # 0.5  # 'Parameter for reg. term in the objective function'
    m=256,  # 'Number of rows in matrix A'
    n=1024,  # 'Number of cols in matrix A'
    data_size=64,  # 'Number of num_samples'
    dist='normal',  # 'Distribution of entries in the matrix A'
    sparsity=0,  # 8  # 'Sparsity' # 非零组数
    gLen=8,  # 16  # 'Length of each group
    data_seed=1,  # 'Seed for generating data'
    logger=None,
    plot_figs=True,
    noise_params=NoiseParams(),
    algorithm='GROUPIMTC',
    repeat_times=1,
    joint_sparse_config=JointSparseConfig(is_joint_sparse=False, mode=SepSparsityType.PERCENTAGE, p=0.8),
    dynamic_settings=[Munch.fromDict({'sparsity': i}) for i in range(10,30 )]
)
# success_rate_exp_opts.sep_rate = int(10 * success_rate_exp_opts.joint_sparse_config.p)

# sub_prox_ =ProxCapped1over2(fixparam=1.0) # #ProxCappedL1(fix_param=1.0)
#sub_prox_ = ProxSCAD(fix_param1=1,fix_param2=3.7)#ProxCappedL1(fix_param=1.0)#ProxTransformedl1(fix_param1=1.0)#ProxMCP(fix_params=3.7)  # ProxSCAD(fix_param1=1,fix_param2=3.7)#ProxCappedL1(fix_param=1.0)
# r'$L_{2,1}$',
# r'$L_{2,\frac{1}{2}}$',
# r'$L_{2,\frac{2}{3}}$',
# r'$L_{2,SCAD}$',
# r'$L_{2,MCP}$',
# r'$L_{2,LOG}$',
# r'$L_{2,Arctan}$',
# r'$L_{2,TL1}$',
# r'$L_{2,CL1}$',
# r'$L_{2,CL1/2}$',
opts = success_rate_exp_opts
sub_prox_ =ProxSCAD(fix_param1=1,fix_param2=3.7)#ProxL2_2over3(opts.n, opts.gLen)#ProxL2_Arctan(opts.n, opts.gLen,c=2)#ProxL2_LogSum(opts.n, opts.gLen,epsilon=1)#ProxL2_1over2(opts.n, opts.gLen)
prox_func = GeneralProxL2Psi(opts.n, opts.gLen,
                             subvec_prox=sub_prox_)  # ProxL2_Arctan(opts.n, opts.gLen, c=1)  # GeneralProxL2Psi(opts.n, opts.gLen, subvec_prox=sub_prox_)
# prox_func =(opts.n, opts.gLen,c=2)
success_rate_exp_opts.save_dir = f'./success_rate_exp_202507/glen8/{prox_func.name()}'


init_log(opts)
reporter = init_settings(opts, None)

for persettings in opts.dynamic_settings:

    opts.sparsity = persettings.sparsity
    opts.logger('\nUsing sparsity: {}\n'.format(opts.sparsity))

    # Create data
    (x_test, d_test), A, b = create_sc_dataset(opts=opts)
    gamma = 1 / np.linalg.norm(A, 2) ** 2
    #prox_func = GeneralProxL2Psi(opts.n, opts.gLen,      subvec_prox=sub_prox_)  # ProxL2_Arctan(opts.n, opts.gLen, c=1)  # GeneralProxL2Psi(opts.n, opts.gLen, subvec_prox=sub_prox_)
    #prox_func = ProxL2_1over2(opts.n, opts.gLen)#ProxL2_LogSum(opts.n, opts.gLen,epsilon=1)
    #ProxL2_0(opts.n, opts.gLen)

    desc = opts.algorithm + '_' + prox_func.name()
    opts.logger('\n Running {} with {}...\n'.format(opts.algorithm, prox_func.name()))

    # Create model
    model = BlockModelFactory().create_model(
        model_name=opts.algorithm, A=A, prox_func=prox_func, tau=opts.tau)
    model(d_test, K=opts.K)
    # for i in range(1, opts.K):
    #     loss = objective_val(model.iter_history[i], d_test, x_test, objective='RELATIVE')
    #     #print(loss)
    success_times = objective_val(model.iter_history[-1], d_test, x_test, objective='SUCCESS_TIMES')
    opts.logger(f'Avg Success Times (success rate) {success_times}')

    df = pd.DataFrame([model.iter_history[-1].reshape(-1).T, x_test.reshape(-1).T]).T
    df.columns = ['x_pred', 'x_gt']
    save_df(df, opts.save_dir + f'/sparsity{opts.sparsity}_{prox_func.name()}',
            f"{desc}_{opts.sparsity}")

    reporter.save_inf(f'\n\nSparse Level --> {opts.sparsity / (opts.n / opts.gLen)}\n\n')
    reporter.save_inf(f'sparsity_{opts.sparsity}_success_times{success_times}')
    if success_times == 0:
        break
