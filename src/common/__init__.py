import typing

from common.enum import SepSparsityType

T = typing.TypeVar('T', int, float)


class JointSparseConfig:
    is_joint_sparse: bool
    p: float  # 取1的概率

    def __init__(self, is_joint_sparse: bool, p: float, mode: SepSparsityType):
        assert 0 <= p <= 1.0, 'join_sparse_rate should be in (0, 1)'
        self.is_joint_sparse = is_joint_sparse
        self.p = p
        self.mode = mode
