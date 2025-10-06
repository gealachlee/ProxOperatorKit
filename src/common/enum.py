from enum import Enum


class DistributionType(Enum):
    LAPLACE = 'laplace'
    RAND = 'rand'
    NORMAL = 'normal'


class SepSparsityType(Enum):
    PERCENTAGE = 'percentage'  # 各组百分比稀疏度
    BINOMIAL = 'binomial'  # 全局稀疏度呈二项分布

