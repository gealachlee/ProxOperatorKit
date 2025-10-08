# ** coding: utf-8 **
"""
@Author: Zhihong Li
@date: 2023/11/25
@description: utils for the project.
@version: 2.0
"""
import logging
import warnings
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd

from common import SepSparsityType



def save_df(df: pd.DataFrame, path: str, filename: str):
    return df.to_excel(path + filename + '.xlsx', index=False)

