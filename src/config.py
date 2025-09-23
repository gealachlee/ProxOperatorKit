# ** coding: utf-8 **
"""
@Author: Zhihong Li, Rongrong Lin
@date: 2024/11/25
@description: Configurations for Group Sparse experiment.
@version: 2.0
"""
import os
from typing import Optional, Union
from munch import DefaultMunch

import utils
from common.enum import DistributionType, SepSparsityType
from common import JointSparseConfig




class NoiseParams(DefaultMunch):
    """
    Configurations for noise added.
    """
    loc: int = 0
    scale: float = 1
    sig: float = 0.001
    dist: DistributionType = DistributionType.RAND


class Opts(DefaultMunch):
    description: str = 'Configurations for Group Sparse experiment.'

    # Problem and Solve Settings (Problem dependent)
    K: int = 1000  # Total Iterations
    objective: str = 'Repeat NMSE'#'#RELATIVE  # Loss Function

    tau: float = 0.1  # 0.5  # 'Parameter for reg. term in the objective function'
    m: int = 256  # 'Number of rows in matrix A'
    n: int = 1024  # 'Number of cols in matrix A'

    data_size: int = 16 # 'Number of num_samples'

    # Generate Data Settings
    dist: str = 'normal'  # 'Distribution of entries in the matrix A'

    sparsity: int = 8  # 8  # 'Sparsity' # 组非零数
    gLen: int = 16  # 16  # 'Length of each group

    data_seed: int = 1 # 'Seed for generating data'

    # Logging Settings
    logger = None
    save_dir: str = '../output'  # Save directory

    plot_figs: bool = False  # 'Plot figures'

    noise_params: Optional[NoiseParams] = NoiseParams()
    joint_sparse_config: JointSparseConfig = JointSparseConfig(is_joint_sparse=False, mode=SepSparsityType.BINOMIAL, p=0.9)


def init_log(opts: Union[Opts, DefaultMunch, dict]):
    # Save directory
    if not os.path.isdir(opts.save_dir):
        os.makedirs(opts.save_dir)

    # Logging file
    logger_file = os.path.join(opts.save_dir, 'output.log')
    opts.logger = utils.setup_logger(logger_file)
    opts.logger('Checkpoints will be saved to directory `{}`'.format(opts.save_dir))
    opts.logger('Log file for training will be saved to file `{}`'.format(logger_file))
    opts.logger('Using tau: {}'.format(opts.tau))  # Output the tau used in current exp
