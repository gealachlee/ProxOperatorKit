from common.config import Settings, JointSparseConfig, NoiseConfig, LogConfig
from common.enum import  SepSparsityType
from prox import ProximalOperator
from prox.group_prox.general import GeneralProxL2Psi
from prox.container import ProximalContainer
from report import Logger
from experiment.experiment import SuccessRateExperiment
from figure_generater.plot_config import PlotConfig


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
    joint_sparse_config=JointSparseConfig(is_joint_sparse=True, mode=SepSparsityType.PERCENTAGE, p=0.5),
)


logger = Logger(**opts.log_config.model_dump()).logger

container = ProximalContainer(
    n=opts.n,
    gLen=opts.gLen,
    data_size=opts.data_size

)

l1_psi_prox_list: list[ProximalOperator] = [
    container.prox_1_1over2(),
    container.prox_1_2over3(),
    container.prox_1_mcp(),
    container.prox_1_scad(),
    container.prox_1_tl1()
]


plot_cfg = PlotConfig('../figure_generater/plot_config_compare_exp.json')


if __name__ == '__main__':
    exp= SuccessRateExperiment()
    result=exp.run(
        logger, opts,
            model_prox_dict=
            {'GROUPPGAC':
            [container.prox_1_1over2()]
             },
    sparsity_scope=list(range(11,15)),model_name='GROUPPGAC'
    )
    exp.plot(plot_cfg, result)
