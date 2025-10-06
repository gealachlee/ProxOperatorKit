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


def init_log(opts: Union[DefaultMunch, dict]):
    # Save directory
    if not os.path.isdir(opts.save_dir):
        os.makedirs(opts.save_dir)

    # Logging file
    logger_file = os.path.join(opts.save_dir, 'output.log')
    opts.logger = utils.setup_logger(logger_file)
    opts.logger('Checkpoints will be saved to directory `{}`'.format(opts.save_dir))
    opts.logger('Log file for training will be saved to file `{}`'.format(logger_file))
    opts.logger('Using tau: {}'.format(opts.tau))  # Output the tau used in current exp
