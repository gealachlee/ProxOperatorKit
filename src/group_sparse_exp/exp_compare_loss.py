# ** coding: utf-8 **
"""
@Author: Zhihong Li, Rongrong Lin
@date: 2024/11/25
@description: Compare Algorithms and Proximal Operators.
@version: 2.0
"""
from common import SepSparsityType
from experiment.experiment import MSELossExperiment
from prox import ProximalOperator
from prox.container import ProximalContainer
from common.config import Settings, JointSparseConfig, NoiseConfig, LogConfig
from figure_generater.plot_config import PlotConfig
from report import Logger

opts = Settings(
    K=1000,
    objective='Repeat NMSE',
    tau=0.1,
    m=256,
    n=1024,
    data_size=1,
    dist='normal',
    gLen=16,
    data_seed=6,
    sparsity=6,
    log_config=LogConfig(file_dir='../', file_name='exp_compare_loss.log', file_mode='a+'),
    plot_figs=True,
    noise_params=NoiseConfig(sig=0.001),
    joint_sparse_config=JointSparseConfig(
        is_joint_sparse=False,
        mode=SepSparsityType.PERCENTAGE,
        p=0.5
    )
)

logger = Logger(**opts.log_config.model_dump()).logger

plot_cfg = PlotConfig('../figure_generater/plot_config_compare_exp.json')

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

mix_psi_prox_list: list[tuple[ProximalOperator, ProximalOperator]] = [
    (container.prox_l0(), container.prox_2_0()),
    (container.prox_l1(), container.prox_2_1()),
    (container.prox_l1over2(), container.prox_2_1over2()),
    (container.prox_l2over3(), container.prox_2_2over3())
]

l2_psi_prox_list: list[ProximalOperator] = [
   # container.prox_2_0(),
    container.prox_2_1(),
    container.prox_2_1over2(),
    container.prox_2_2over3(),
    container.prox_2_scad(),
    container.prox_2_mcp(),
    container.prox_2_log(),
    container.prox_2_arctan(),
    container.prox_2_tl1(),
    container.prox_2_CL1(),
    container.prox_2_CL1over2()
]

model_prox_dict = {
    'GROUPPGAC': l2_psi_prox_list,  #
}

if __name__ == '__main__':
    exp = MSELossExperiment()
    opts.sparsity = 8
    total_results1 = exp.run(logger, opts, model_prox_dict)
    opts.sparsity = 10
    total_results2 = exp.run(logger, opts, model_prox_dict)
    exp.plot(plot_cfg,total_results1,total_results2)