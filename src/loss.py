import numpy as np


def objective_val(x, d, x_gt, objective: str):
    # Objective function
    if objective == 'Repeat NMSE':
        l2 = ((x - x_gt) ** 2).sum(axis=1)
        denom = (x_gt ** 2).sum(axis=1)
        val = (l2 / denom).mean()
    elif objective == 'GT':
        val = ((x - x_gt) ** 2).sum(dim=1).mean()
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
        val=(val < 0.005).mean()
        print(f'val={val}')
    else:
        raise ValueError('Invalid objective option {}'.format(objective))

    return val
