import pandas as pd
from munch import DefaultMunch
from munch import Munch

from common import JointSparseConfig
from common.enum import SepSparsityType
from config import NoiseParams
from config import init_log
from dataset.create_data import create_sc_dataset
from loss import objective_val
from models import BlockModelFactory
from prox.group_prox import ProxL1_1over2, ProxL1_2over3
from prox.group_prox.l1psi import *
from prox.group_prox.l2psi.prox_cl import *
from prox.sep_prox.prox_cl import *
from report import init_settings

success_rate_exp_opts = DefaultMunch(
    K=1000,  # Total Iterations
    objective='Repeat NMSE',
    tau=0.1,  # 0.5  # 'Parameter for reg. term in the objective function'
    m=256,  # 'Number of rows in matrix A'
    n=1024,  # 'Number of cols in matrix A'
    data_size=64,  # 'Number of num_samples'
    dist='normal',  # 'Distribution of entries in the matrix A'
    sparsity=8,  # 8  # 'Sparsity' # 非零组数
    gLen=8,  # 16  # 'Length of each group
    data_seed=None,  # 'Seed for generating data'
    logger=None,
    plot_figs=True,
    noise_params=NoiseParams(),
    algorithm='GROUPIMTC',
    repeat_times=1,
    joint_sparse_config=JointSparseConfig(is_joint_sparse=True, mode=SepSparsityType.PERCENTAGE, p=0.5),
    dynamic_settings= [Munch.fromDict({'sparsity': i}) for i in range(16, 31)]
)
success_rate_exp_opts.sep_rate = int(10 * success_rate_exp_opts.joint_sparse_config.p)
success_rate_exp_opts.save_dir = f'./exp/success_rate/{success_rate_exp_opts.joint_sparse_config.mode.value}/'

opts = success_rate_exp_opts

init_log(opts)
reporter = init_settings(opts, None)

def run():
    for persettings in opts.dynamic_settings:
        success_times = 0
        opts.sparsity = persettings.sparsity
        opts.logger('\nUsing sparsity: {}\n'.format(opts.sparsity))
       # for repeat_times in range(1, opts.repeat_times + 1):
            # Create data
        (x_test, d_test), A, b = create_sc_dataset(opts=opts)
        gamma = 1 / np.linalg.norm(A, 2) ** 2
        prox_func = ProxL1_1over2(opts.n, opts.gLen)

        desc = opts.algorithm + '_' + prox_func.name()
        opts.logger('\n Running {} with {}...\n'.format(opts.algorithm, prox_func.name()))

            # Create model
        model = BlockModelFactory().create_model(
                model_name=opts.algorithm, A=A, prox_func=prox_func, tau=opts.tau)
        model(d_test, K=opts.K)

        loss = objective_val(model.iter_history[-1], d_test, x_test, objective='SUCCESS_TIMES')
        opts.logger('Iteration: {}, Testing Loss: {}'.format(opts.K, loss))
           # if loss < 0.005:
            #     success_times += 1
        df = pd.DataFrame([model.iter_history[-1].reshape(-1).T, x_test.reshape(-1).T]).T
        df.columns = ['x_pred', 'x_gt']
            # save_df(df, opts.save_dir + f'/sparsity{opts.sparsity}_repeat{repeat_times}.xlsx',
            #         f"{desc}_{opts.sparsity}_{repeat_times}")
        setattr(opts, f'sparsity_{opts.sparsity}_success_times', success_times)
        setattr(opts, f'sparsity_{opts.sparsity}_success_rate', success_times / opts.repeat_times)
        reporter.save_inf(f'\n\nSparse Level --> {opts.sparsity / (opts.n / opts.gLen)}\n\n')
        reporter.save_inf(f'sparsity_{opts.sparsity}_success_times{success_times}')
        reporter.save_inf(f'sparsity_{opts.sparsity}_success_rate{success_times / opts.repeat_times}')
        reporter.save_inf(f'\n\nFinish -sparsity-{opts.sparsity}-repeat-{opts.repeat_times}\n\n')

run()

