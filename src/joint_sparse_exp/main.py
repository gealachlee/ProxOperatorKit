from munch import DefaultMunch

from common import JointSparseConfig, SepSparsityType
from config import NoiseParams, init_log
from dataset.create_data import create_sc_dataset
from loss import objective_val
from models import JointModelFactory, BlockModelFactory
from prox.group_prox import ProxL1_1over2, ProxL1_2over3
from prox.group_prox.l1psi import *
from prox.group_prox.l2psi.prox_cl import *
from prox.sep_prox.prox_cl import *
from report import init_settings
import matplotlib
matplotlib.use('TkAgg')

opts = DefaultMunch(
    K=1000,  # Total Iterations
    objective='SUCCESS_TIMES',
    tau=0.05,  # 0.5  # 'Parameter for reg. term in the objective function'
    m=256,  # 'Number of rows in matrix A'
    n=1024,  # 'Number of cols in matrix A'
    data_size=64,  # 'Number of num_samples'
    dist='normal',  # 'Distribution of entries in the matrix A'
    gLen=8,  # 16  # 'Length of each group
    data_seed=1,  # 'Seed for generating data'
    logger=None,
    sparsity=12,
    plot_figs=True,
    noise_params=NoiseParams(),
    joint_sparse_config=JointSparseConfig(is_joint_sparse=True, mode=SepSparsityType.PERCENTAGE, p=0.8),
)
opts.save_dir = f'./exp/20250921'

init_log(opts)
reporter = init_settings(opts, 'different_L1psi')

# ---- Joint Proximal Operators
gamma = 1

prox_1_1over2 = ProxL1_1over2(opts.n, opts.gLen, opts.data_size)
prox_1_2over3 = ProxL1_2over3(opts.n, opts.gLen, opts.data_size)

prox_1_mcp = GeneralProxL1Psi(opts.n, opts.gLen,
                              subvec_prox=L1_MCP_SubvecProx(
                                  opts.gLen, lamb=0, fix_param=3.7,
                                  precompute=False),
                              num_samples=opts.data_size)

prox_1_scad = GeneralProxL1Psi(opts.n, opts.gLen,
                               subvec_prox=L1_SCAD_SubvecProx(
                                   opts.gLen, lamb=0, fix_param1=1,
                                   fix_param2=3.7,
                                   precompute=False),
                               num_samples=opts.data_size)
prox_1_tl1 = GeneralProxL1Psi(opts.n, opts.gLen,
                              subvec_prox=L1_TransformedL1_SubvecProx(
                                  opts.gLen, lamb=0, fix_param=4,
                                  precompute=False),
                              num_samples=opts.data_size)
# Joint
# Joint
prox_2over3 =  ProxL2over3()
prox_1 = ProxL1()
prox_0 = ProxL0()
prox_1over2 =  ProxL1over2()
#prox_2_1over2 = ProxL2_1over2(opts.n, opts.gLen)



prox_2_0 = ProxL2_0(opts.n, opts.gLen)
prox_2_2over3 = ProxL2_2over3(opts.n, opts.gLen)
prox_2_1 = ProxL2_1(opts.n, opts.gLen)
prox_2_1over2 = ProxL2_1over2(opts.n, opts.gLen)

experiment_prox = [prox_1_1over2, prox_1_2over3, prox_1_mcp, prox_1_scad, prox_1_tl1]
model_prox_dict = {
   # 'JointFISTA': [(prox_1,prox_2_1)],#experiment_prox,  #
     'IMTC':[(prox_1over2,prox_2_1over2)]#,(prox_1,prox_2_1),(prox_1over2,prox_2_1over2),(prox_2over3,prox_2_2over3)],
   # 'GROUPIMTC':[prox_1_scad]#experiment_prox#experiment_prox#experiment_prox#,

}

# {\color{red} $L_{2,|\cdot|}+\ell_1$ and $L_{2,|\cdot|_0}+\ell_0$}

name_list = [
   # r'$L_{2,|\cdot|_0}+\ell_0$',
    # r'$L_{2,|\cdot|}+\ell_1$',
    # r'$L_{2,|\cdot|^{1/2}}+\ell_{1/2}$',
     r'$L_{2,|\cdot|^{2/3}}+\ell_{2/3}$',
   #
   #  r'$L_{1,|\cdot|^{1/2}}$',
   #  r'$L_{1,|\cdot|^{2/3}}$',
   #  r'$L_{1,{\rm MCP}}$',
   # r'$L_{1,{\rm SCAD}}$',
   #  r'$L_{1,{\rm TL1}}$'
]
# init_log(opts)
def main(opts):
    print('\nSparsity: {}\n'.format(opts.sparsity))
    save_model = []
    total_results = DefaultMunch()
    for model_name, prox_func_list in model_prox_dict.items():
            for prox_func in prox_func_list:
                for per_sp in range(20, 35):
                    opts.sparsity = per_sp
                    opts.logger('\nUsing sparsity: {}\n'.format(opts.sparsity) + f'{model_name}')
                    (x_test, d_test), A, b = create_sc_dataset(opts=opts)
                    if not isinstance(prox_func, tuple):
                        opts.logger('\n Running {} with {}...\n'.format(model_name, prox_func.name()))
                        desc = model_name + '_' + prox_func.name()
                        total_results.__setitem__(desc, [])
                        model = BlockModelFactory().create_model(
                            model_name=model_name, A=A, prox_func=prox_func, tau=opts.tau)

                    else:
                        opts.logger(
                            '\n Running {} with {} and {}...\n'.format(model_name, prox_func[0].name(), prox_func[1].name()))
                        desc = model_name + '_' + prox_func[0].name() + '_' + prox_func[1].name()
                        total_results.__setitem__(desc, [])
                        model = JointModelFactory().create_model(
                            model_name=model_name, A=A, prox_func1=prox_func[0], prox_func2=prox_func[1], tau=opts.tau)
                    model(d_test, K=opts.K)

                # save_history[f'{rep}_{model_name}'] = model.iter_history
                    save_model.append(model)

                    loss_= objective_val(model.iter_history[-1], d_test, x_test, objective=opts.objective).item()
                    #opts.logger(f'{prox_func.name()}--Sparsity:{per_sp}---Success Rate:{loss_}')
                    opts.logger(f'{prox_func[0].name()}--{prox_func[1].name()}--Sparsity:{per_sp}---Success Rate:{loss_}')

                # for k in range(0, opts.K):
                #     test_loss =
                #     testing_loss = np.mean(test_loss)  # Compute the average of the losses
                #     total_results[desc].append(testing_loss)
                  #  opts.logger('Iteration: {}, Testing Loss: {}'.format(k, testing_loss))
                    if loss_==0:
                        break
            print(f'{model_name}:-->Experiment: Sparsity={opts.sparsity},gLen={opts.gLen},sparsity={per_sp}')


    return total_results


print(f'Experiment: Sparsity={opts.sparsity},gLen={opts.gLen}')
total_results1 = main(opts)
