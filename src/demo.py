# ** coding: utf-8 **
"""
@Author: Zhihong Li
@date: 2025/09/25
@description: demo for the project, including the main function.
@version: 4.0
"""
from common import SepSparsityType
from common.config import Settings, JointSparseConfig, NoiseConfig, LogConfig
from dataset.create_data import create_sc_dataset
from experiment.experiment import MSELossExperiment
from prox import ProximalOperator
from prox.container import ProximalContainer
from report import Logger

opts = Settings(
    K=400,
    objective='Repeat NMSE',
    tau=0.02,
    m=256,
    n=1024,
    data_size=16,
    dist='normal',
    gLen=1,
    data_seed=6,
    sparsity=30,
    plot_figs=True,
    log_config=LogConfig(file_dir='./', file_name='demo.log', file_mode='w+'),
    noise_params=NoiseConfig(sig=0.001),
    joint_sparse_config=JointSparseConfig(
        is_joint_sparse=False,
        mode=SepSparsityType.PERCENTAGE,
        p=0.5
    )
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

mix_psi_prox_list: list[tuple[ProximalOperator, ProximalOperator]] = [
    (container.prox_l0(),container.prox_2_0()),
    (container.prox_l1(),container.prox_2_1()),
    (container.prox_l1over2(),container.prox_2_1over2()),
    (container.prox_l2over3(),container.prox_2_2over3())
]

cl_prox_list :list= [
    container.prox_l1(),
    container.prox_cl1(),  # blue
    container.prox_l1over2(), # red
    container.prox_cl1over2() #  green
]
l2_psi_prox_list: list[ProximalOperator] = [
    container.prox_2_0(),
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

if __name__=="__main__":
    model_prox_dict = {
        'FISTA': cl_prox_list
        # 'IMTC': l1_psi_prox_list
    }
    (x_test, d_test), A, b = create_sc_dataset(opts=opts)

    exp=MSELossExperiment()

    total_results = exp.run(logger,opts, model_prox_dict)
    res= total_results.results
    import matplotlib.pyplot as plt

    #
    # from figure_generater.plot_config import PlotConfig
    # plot_cfg = PlotConfig(json_files='./figure_generater/plot_config_compare_exp.json')
    # plot_color=plot_cfg.plot_color
    # fig, (ax1) = plt.subplots(1, 1, figsize=(16, 6))
    # fig.subplots_adjust(hspace=0.5)  #
    # for index, record in enumerate(total_results.results):
    #     linestyle = '--' if 'Joint' in record.desc else '-'
    #     ax1.plot(record.metrics, color=plot_color[index], linestyle=linestyle)
    #     ax1.set_yscale('log')
    #     ax1.set_yticks([10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4],
    #                    [r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$'])
    #
    # ax1.set_ylabel('Relative Error', fontdict={'fontsize': 12})
    # ax1.set_xlabel('Iteration', fontdict={'fontsize': 12})
    # plt.legend()
    # plt.savefig('1.png')

    logger.info(f'{res[0].desc}--{res[0].metrics[-1]}')
    logger.info(f'{res[1].desc}--{res[1].metrics[-1]}')
    logger.info(f'{res[2].desc}--{res[2].metrics[-1]}')
    logger.info(f'{res[3].desc}--{res[3].metrics[-1]}')