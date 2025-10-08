from common.config import Settings, JointSparseConfig, NoiseConfig, LogConfig
from common.enum import  SepSparsityType
from prox.group_prox.general import GeneralProxL2Psi
from prox.container import ProximalContainer
import pandas as pd
from report import Logger
from experiment.experiment import SuccessRateExperiment

opts = Settings(
    K=1000,  # Total Iterations
    objective='SUCCESS_TIMES',
    tau=0.05,  # 0.5  # 'Parameter for reg. term in the objective function'
    m=256,  # 'Number of rows in matrix A'
    n=1024,  # 'Number of cols in matrix A'
    data_size=64,  # 'Number of num_samples'
    dist='normal',  # 'Distribution of entries in the matrix A'
    sparsity=1,
    gLen=8,  # 16  # 'Length of each group
    data_seed=1,  # 'Seed for generating data'
    log_config=LogConfig(file_dir='./',file_name='success_rate_exp.log',is_debug=True),
    plot_figs=True,
    noise_params=NoiseConfig(sig=0.001),
    joint_sparse_config=JointSparseConfig(is_joint_sparse=False, mode=SepSparsityType.PERCENTAGE, p=0.8),
)

logger = Logger(**opts.log_config.model_dump()).logger

if __name__ == '__main__':
    exp= SuccessRateExperiment()
    result=exp.run(
        logger, opts,
            model_prox_dict=
            {'GROUPPGAC':
            [GeneralProxL2Psi(opts.n, opts.gLen,
                subvec_prox=ProximalContainer.prox_scad(fix_param1=12, fix_param2=3.7))]
             },
    sparsity_scope=list(range(11,30))
    )