# ** coding: utf-8 **
"""
@Author: Zhihong Li, Rongrong Lin
@date: 2024/11/25
@description: Compare Algorithms and Proximal Operators.
@version: 2.0
"""
import time

from munch import DefaultMunch
from config import init_log
from dataset.create_data import create_sc_dataset
from joint_sparse_exp.exp_settings import exp2_opts
from loss import objective_val
from models import JointModelFactory, BlockModelFactory
from prox.group_prox import ProxL2_1
from prox.group_prox.l1psi_dev import *
from prox.group_prox.prox_cl import *

from prox.sep_prox.prox_cl import *
from report import init_settings

# initial log
opts = exp2_opts
reporter = init_settings(opts, 'different_sep_sparsity_levels')

init_log(opts)
name_list = ['prox_2_1', 'prox_2_tl1', 'prox_2_1over2', 'prox_2_2over3', 'prox_2_arctan', 'prox_2_mcp', 'prox_2_scad',
             'prox_2_CL1', 'prox_2_CL1over2'
             ]

save_model = []
total_results = DefaultMunch()

(x_test, d_test), A, b = create_sc_dataset(opts=opts)
gamma = 1 / np.linalg.norm(A, 2) ** 2
# ---- Element wise Proximal Operators
# prox_l1 = ProxL1()
# prox_1over2 = ProxL1over2()
# prox_2over3 = ProxL2over3()
# prox_tl1 = ProxTransformedl1(fix_param1=1)
# prox_mcp = ProxMCP(fix_params=3.7)
# prox_scad = ProxSCAD(fix_param1=1, fix_param2=3.7)
prox_cl1 = ProxCappedL1(fix_param=1)
prox_cl1over2 = ProxCapped1over2(fixparam=1)

# ---- Group Proximal Operators
prox_2_1 = ProxL2_1(opts.n, opts.gLen)
# prox_2_2over3 = ProxL2_2over3(opts.n, opts.gLen)
# prox_2_1over2 = ProxL2_1over2(opts.n, opts.gLen)
# prox_2_arctan = ProxL2_Arctan(opts.n, opts.gLen, c=1)
# prox_2_tl1 = GeneralProxL2Psi(opts.n, opts.gLen, prox_tl1)
# prox_2_mcp = GeneralProxL2Psi(opts.n, opts.gLen, prox_mcp)
# prox_2_scad = GeneralProxL2Psi(opts.n, opts.gLen, prox_scad)
# prox_2_CL1 = GeneralProxL2Psi(opts.n, opts.gLen, prox_cl1)
# prox_2_CL1over2 = GeneralProxL2Psi(opts.n, opts.gLen, prox_cl1over2)

# ---- Joint Proximal Operators
# prox_1_1over2_Fixed = ProxL1_1over2_FixParams(opts.n, opts.gLen, opts.data_size, nu=opts.tau * gamma)
# prox_1_1over2 = ProxL1_1over2(opts.n, opts.gLen, opts.data_size)
# prox_1_2over3_Fixed = ProxL1_2over3_FixParams(opts.n, opts.gLen, opts.data_size, nu=opts.tau * gamma)
# prox_mcp = GeneralProxL1Psi(opts.n, opts.gLen,
#                             subvec_prox=L1_MCP_SubvecProx(opts.gLen, lamb=opts.tau * gamma, fix_param=3.7,
#                                                           precompute=True),
#                             num_samples=opts.data_size)
# prox_1_SCAD = GeneralProxL1Psi(opts.n, opts.gLen,
#                                subvec_prox=L1_SCAD_SubvecProx(opts.gLen, lamb=opts.tau * gamma, fix_param1=1.0,
#                                                               fix_param2=3.7, precompute=False),
#                                num_samples=opts.data_size)
# prox_1_tl1 = GeneralProxL1Psi(opts.n, opts.gLen,
#                               subvec_prox=L1_TransformedL1_SubvecProx(opts.gLen, lamb=opts.tau * gamma, fix_param=2,
#                                                                       precompute=False),
#                               num_samples=opts.data_size)
prox_1_1over2 = ProxL1_1over2_FixParams(opts.n, opts.gLen, opts.data_size, nu=opts.tau * gamma)
prox_1_2over3 = ProxL1_2over3_FixParams(opts.n, opts.gLen, opts.data_size, nu=opts.tau * gamma)

# scad = ProxSCAD(fix_param1=1.0,fix_param2=3.7)

model_prox_dict = {
    'FISTA': [prox_1_2over3],  #

}

t = []


def main():
    for model_name, prox_func_list in model_prox_dict.items():
        for prox_func in prox_func_list:

            start = time.time()
            if not isinstance(prox_func, tuple):
                opts.logger('\n Running {} with {}...\n'.format(model_name, prox_func.name()))
                desc = model_name + '_' + prox_func.name()
                total_results.__setitem__(desc, [])
                model = BlockModelFactory().create_model(
                    model_name=model_name, A=A, prox_func=prox_func, tau=opts.tau)

            else:
                raise NotImplementedError
                exit(1)
                opts.logger(
                    '\n Running {} with {} and {}...\n'.format(model_name, prox_func[0].name(), prox_func[1].name()))
                desc = model_name + '_' + prox_func[0].name() + '_' + prox_func[1].name()
                total_results.__setitem__(desc, [])
                model = JointModelFactory().create_model(
                    model_name=model_name, A=A, prox_func1=prox_func[0], prox_func2=prox_func[1], tau=opts.tau)
            model(d_test, K=opts.K)

            # save_history[f'{rep}_{model_name}'] = model.iter_history
            save_model.append(model)
            opts.logger('Testing losses:')
            for k in range(0, opts.K):
                test_loss = objective_val(model.iter_history[k], d_test, x_test, objective=opts.objective).item()
                testing_loss = np.mean(test_loss)  # Compute the average of the losses
                total_results[desc].append(testing_loss)
                opts.logger('Iteration: {}, Testing Loss: {}'.format(k, testing_loss))
            end=time.time()
            print(end - start)
            t.append(end-start)


main()

filename = f'sparsity{opts.sparsity}_gLen{opts.gLen}_sep_rate{opts.sep_rate}'

################
# plot fig
import matplotlib.pyplot as plt
import matplotlib

plot_color = [
    "b",  # 蓝色
    "r",  # 红色
    "g",  # 绿色
    "c",  # 青色
    "m",  # 品红
    "y",  # 黄色
    "k",  # 黑色
    "#FFA500",  # 橙色
    "#800080",  # 紫色
    "#FFD700",  # 金色
    "#00FF7F",  # 春绿色
    "#FF4500",  # 橙红色
    "#1E90FF",  # 道奇蓝
    "#8A2BE2",  # 蓝紫色
    "#FF1493"  # 深粉色
]

for index, (name, loss) in enumerate(total_results.items()):
    linestyle = '--' if 'Joint' in name else '-'
    plt.plot(loss, color=plot_color[index], label=name_list[index], linestyle=linestyle)
    plt.yscale('log')
    plt.yticks([10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4, 10 ** -5],
               [r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$'],
               fontsize=8)
    plt.legend()
plt.ylabel('Relative error')
plt.xlabel('Iterations')

if opts.save_dir is not None:
    plt.savefig(opts.save_dir + f'/{filename}.png', dpi=600)


# plt.show()


# if opts.save_dir is not None:
#     plt.savefig(opts.save_dir + '/plot.png',dpi=800)
# plt.show()

def save_results_xlsx(filename):
    import pandas as pd
    final_output = []
    for permodel in save_model:
        final_output.append(permodel.iter_history[-1].reshape(-1))
    final_output.append(x_test.reshape(-1))
    names = [i.name() for i in save_model]
    names.append('standard')
    df = pd.DataFrame(final_output).T
    df.columns = names
    df.to_excel(f'{opts.save_dir}/{filename}.xlsx')

# save_results_xlsx(f'{filename}')
