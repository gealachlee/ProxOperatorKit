# ** coding: utf-8 **
"""
@Author: Zhihong Li, Rongrong Lin
@date: 2024/11/25
@description: Compare Algorithms and Proximal Operators.
@version: 2.0
"""

from munch import DefaultMunch

from common import JointSparseConfig, SepSparsityType
from config import init_log, NoiseParams
from dataset.create_data import create_sc_dataset
from joint_sparse_exp.exp_settings import exp2_opts
from loss import objective_val
from models import JointModelFactory, BlockModelFactory
from prox.group_prox.general import GeneralProxL2Psi
from prox.group_prox.l1psi_dev import *
from prox.group_prox.prox_cl import *
from prox.group_prox.l2psi.prox_cl import *
from prox.sep_prox.prox_cl import *
from report import init_settings

opts=DefaultMunch(
    K=1000,  # Total Iterations
    objective='Repeat NMSE',
    tau=0.1,  # 0.5  # 'Parameter for reg. term in the objective function'
    m=256,  # 'Number of rows in matrix A'
    n=1024,  # 'Number of cols in matrix A'
    data_size=1,  # 'Number of num_samples'
    dist='normal',  # 'Distribution of entries in the matrix A'
    gLen=16,  # 16  # 'Length of each group
    data_seed=1,#6,  # 'Seed for generating data'
    logger=None,
    sparsity=9,
    plot_figs=True,
    noise_params=NoiseParams(),
    joint_sparse_config=JointSparseConfig(is_joint_sparse=False, mode=SepSparsityType.PERCENTAGE, p=0.2),
)
opts.save_dir = f'./20250808/exp/'
reporter = init_settings(opts, '20250808/different_L2psi')

init_log(opts)
name_list = [
    r'$L_{2,|·|_0}$',
    r'$L_{2,|\cdot|}$',
    r'$L_{2,|\cdot|^{1/2}}$',
    r'$L_{2,|\cdot|^{2/3}}$',
    r'$L_{2,{\rm SCAD}}$',
    r'$L_{2,{\rm MCP}}$',
    r'$L_{2,{\rm LOG}}$',
    r'$L_{2,{\rm Arctan}}$',
    r'$L_{2,{\rm TL1}}$',
    r'$L_{2,{\rm CL1}}$',
    r'$L_{2,{\rm CL1/2}}$',
]



# ---- Element wise Proximal Operators
prox_l1 = ProxL1()
prox_1over2 = ProxL1over2()
prox_2over3 = ProxL2over3()
prox_tl1 = ProxTransformedl1(fix_param1=1)
prox_mcp = ProxMCP(fix_params=3.7)
prox_scad = ProxSCAD(fix_param1=1, fix_param2=3.7)# a 尽可能小）
prox_cl1 = ProxCappedL1(fix_param=1)
prox_cl1over2 = ProxCapped1over2(fixparam=1)

# ---- Group Proximal Operators
prox_2_0 = ProxL2_0(opts.n, opts.gLen)
prox_2_1 = ProxL2_1(opts.n, opts.gLen)
prox_2_1over2 = ProxL2_1over2(opts.n, opts.gLen)
prox_2_2over3 = ProxL2_2over3(opts.n, opts.gLen)
prox_2_scad = GeneralProxL2Psi(opts.n, opts.gLen, prox_scad)
prox_2_mcp = GeneralProxL2Psi(opts.n, opts.gLen, prox_mcp)
prox_2_log = ProxL2_LogSum(opts.n, opts.gLen,epsilon=1)
prox_2_arctan = ProxL2_Arctan(opts.n, opts.gLen, c=2)
prox_2_tl1 = GeneralProxL2Psi(opts.n, opts.gLen, prox_tl1)
prox_2_CL1 = GeneralProxL2Psi(opts.n, opts.gLen, prox_cl1)
prox_2_CL1over2 = GeneralProxL2Psi(opts.n, opts.gLen, prox_cl1over2)

experiment_prox = [prox_2_0,prox_2_1, prox_2_1over2, prox_2_2over3, prox_2_scad,
                   prox_2_mcp, prox_2_log, prox_2_arctan, prox_2_tl1, prox_2_CL1, prox_2_CL1over2]

model_prox_dict = {
    'GROUPIMTC': experiment_prox,  #
}


def main(opts):
    print('\nSparsity: {}\n'.format(opts.sparsity))
    save_model = []
    total_results = DefaultMunch()
    for model_name, prox_func_list in model_prox_dict.items():
        (x_test, d_test), A, b = create_sc_dataset(opts=opts)

        for prox_func in prox_func_list:
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
    return total_results

opts.sparsity=8
print(f'Experiment: Sparsity={opts.sparsity},gLen={opts.gLen}')
total_results1=main(opts)
opts.sparsity=10
print(f'Experiment: Sparsity={opts.sparsity},gLen={opts.gLen}')
total_results2=main(opts)
filename = f'sparsity{opts.sparsity}_gLen{opts.gLen}_sep_rate{opts.sep_rate}'

# plot fig
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(hspace=0.5)  #
for index, (name, loss) in enumerate(total_results1.items()):
    linestyle = '--' if 'Joint' in name else '-'
    ax1.plot(loss, color=plot_color[index], label=name_list[index], linestyle=linestyle)
    ax1.set_yscale('log')
    ax1.set_yticks([10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4, 10 ** -5, 10 ** -6, 10 ** -7],
                   [r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$',r'$10^{-6}$',r'$10^{-7}$'])

ax1.set_ylabel('Relative Error',fontdict={'fontsize': 12})
ax1.set_xlabel('Iteration',fontdict={'fontsize': 12})


for index, (name, loss) in enumerate(total_results2.items()):
    linestyle = '--' if 'Joint' in name else '-'
    ax2.plot(loss, color=plot_color[index], label=name_list[index], linestyle=linestyle)
    ax2.set_yscale('log')
    ax2.set_yticks([10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4,10**-5, 10 ** -6, 10 ** -7],
               [r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$',r'$10^{-6}$',r'$10^{-7}$'])
ax2.set_ylabel('Relative Error',fontdict={'fontsize': 12})
ax2.set_xlabel('Iteration',fontdict={'fontsize': 12})
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.51, 0.97), ncol=11,fontsize=12)

# 显示图形
plt.tight_layout(rect=[0, 0, 1, 0.9])

if opts.save_dir is not None:
    plt.savefig(opts.save_dir + f'/{filename}.png', dpi=300)
plt.show()


# if opts.save_dir is not None:
#     plt.savefig(opts.save_dir + '/plot.png',dpi=800)
# plt.show()

# def save_results_xlsx(filename):
#     import pandas as pd
#     final_output = []
#     for permodel in save_model:
#         final_output.append(permodel.iter_history[-1].reshape(-1))
#     final_output.append(x_test.reshape(-1))
#     names = [i.name() for i in save_model]
#     names.append('standard')
#     df = pd.DataFrame(final_output).T
#     df.columns = names
#     df.to_excel(f'{opts.save_dir}/{filename}.xlsx')

# save_results_xlsx(f'{filename}')
