# ** coding: utf-8 **
"""
@Author: Zhihong Li
@date: 2025/09/25
@description: demo for the project, including the main function.
@version: 4.0
"""
from munch import DefaultMunch
from common import SepSparsityType
from config import init_log
from dataset.create_data import create_sc_dataset

from loss import objective_val
from prox import ProximalOperator
from prox.container import ProximalContainer
import numpy as np
from models import initialize_model
from common.config import Settings, JointSparseConfig, NoiseConfig

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
    logger=None,
    sparsity=6,
    plot_figs=True,
    noise_params=NoiseConfig(sig=0.01),
    joint_sparse_config=JointSparseConfig(
        is_joint_sparse=False,
        mode=SepSparsityType.PERCENTAGE,
        p=0.5
    )
)

init_log(opts)

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

model_prox_dict = {
    'GROUPIMTC': l2_psi_prox_list,
    # 'IMTC': l1_psi_prox_list
}

(x_test, d_test), A, b = create_sc_dataset(opts=opts)

def main(opts):
    print('\nSparsity: {}\n'.format(opts.sparsity))
    save_model = []
    total_results = DefaultMunch()

    for model_name, prox_func_list in model_prox_dict.items():
        for prox_func in prox_func_list:
            model = initialize_model(model_name=model_name, prox_func=prox_func, A=A, opts=opts)
            desc = model_name + '_' + prox_func.name() if not isinstance(prox_func, tuple) else model_name + '_' + \
                                              prox_func[0].name() + '_' + \
                                                                                                prox_func[1].name()
            opts.logger('Model: {}\n'.format(desc))
            total_results.__setitem__(desc, [])
            model(d_test, K=opts.K)

            save_model.append(model)
            opts.logger('Testing losses:')
            for k in range(0, opts.K):
                test_loss = objective_val(model.iter_history[k], d_test, x_test, objective=opts.objective).item()
                testing_loss = np.mean(test_loss)
                total_results[desc].append(testing_loss)
                opts.logger('Iteration: {}, Testing Loss: {}'.format(k, testing_loss))
    return total_results


total_results = main(opts)
