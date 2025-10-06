# ** coding: utf-8 **
"""
@Author: Zhihong Li
@date: 2024/10/25
@description: loss functions for the project.
@version: 1.0
"""

import numpy as np


def calculate_normalized_mean_squared_error(x, x_gt):
    # Calculate Normalized Mean Squared Error (NMSE)
    l2 = ((x - x_gt) ** 2).sum(axis=1)
    denom = (x_gt ** 2).sum(axis=1)
    val = (l2 / denom).mean()
    return val

def calculate_gt(x, x_gt):
    # Calculate GT
    return ((x - x_gt) ** 2).sum(dim=1).mean()

def objective_val(x, d, x_gt, objective: str):
    # Objective function
    if objective == 'Repeat NMSE':
        return calculate_normalized_mean_squared_error(x, x_gt)
    elif objective == 'GT':
        return calculate_gt(x, x_gt)
    elif objective == 'NMSE':
        l2 = ((x - x_gt) ** 2).mean()
        denom = (x_gt ** 2).mean()
        val = 10.0 * np.log10(l2 / denom)
    elif objective == 'RELATIVE':
        l2 = ((x - x_gt) ** 2).mean()
        denom = (x_gt ** 2).mean()
        val = l2 / denom
    elif objective == 'SUCCESS_TIMES':
        l2 = ((x - x_gt) ** 2).sum(axis=1)
        denom = (x_gt ** 2).sum(axis=1)
        val = np.nan_to_num((l2 / denom), nan=0)
        val = (val < 0.005).mean()
    else:
        raise ValueError('Invalid objective option {}'.format(objective))

    return val
