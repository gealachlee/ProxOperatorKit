# ** coding: utf-8 **
import logging
import warnings
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd

from common import SepSparsityType

plot_color = [
    "b",
    "r",
    "c",
    "m",
    "k",
    "g",
    'y'
]
plot_styles = [
    "-",
    ".",
    ":",
    '-',
]


def setup_logger(log_file):
    if log_file is not None:
        logging.basicConfig(filename=log_file, level=logging.INFO)
        lgr = logging.getLogger()
        lgr.addHandler(logging.StreamHandler())
        lgr = lgr.info
    else:
        lgr = print

    return lgr


def plot_linechart(loss_dict: Dict[str, List[float]], save_dir: str = None, figname: str = '/plot.png', **kwargs):
    for index, (name, loss) in enumerate(loss_dict.items()):
        # color = random.choice(plot_color)
        # style= random.choice(plot_styles)

        plt.plot(loss, color=plot_color[index], label=name)
        # plt.yscale('log')
        # plt.yticks([10**0,10 ** -1, 10 ** -2, 10 ** -3],
        #            [r'$10^{0}$',r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$'])
        plt.legend()
    plt.title(kwargs.get('title', ''))
    plt.show()
    if save_dir is not None:
        plt.savefig(save_dir + figname)


def save_df(df: pd.DataFrame, path: str, filename: str):
    return df.to_excel(path + filename + '.xlsx', index=False)


def describe_total_sparsity(opts, reporter):
    """
    Calculate
    inter-group sparsity and intra-group sparsity

    """
    inter_group_sparsity = opts.sparsity / (opts.n / opts.gLen)
    print(f'\n\nSparse Level --> {inter_group_sparsity}\n\n')
    reporter.save_inf(f'\n\nSparse Level (inter-group-sparsity) --> {inter_group_sparsity}\n\n')

    if opts.joint_sparse_config.is_joint_sparse:
        if opts.joint_sparse_config.mode == SepSparsityType.BINOMIAL:
            warnings.warn('intra_group_sparsity is not defined because of binomial distribution')
        else:
            intra_group_sparsity = opts.joint_sparse_config.p
            print(f'\n\n intra-group sparsity {intra_group_sparsity}\n\n')
            reporter.save_inf(f'\n\n intra-group sparsity {intra_group_sparsity}\n\n')

            # ratio of the intra-group sparsity to inter-group sparsity
            ratio = intra_group_sparsity * opts.gLen / opts.sparsity
            print(f'\n\n ratio {ratio} \n\n')
            reporter.save_inf(f'\n\n ratio of the intra-group sparsity to inter-group sparsity {ratio}\n\n')
