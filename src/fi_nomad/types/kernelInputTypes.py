"""Defines types for objects passed to kernels in instantiation. Kernel-specific parameter
sets will also be included here.

Classes:
    KernelInputType: Standard data object for initializing kernels.
    AggressiveMomentumAdditionalParameters: Data object containing algorithm-specific
        parameters.

"""

from typing import NamedTuple, Union
from .types import FloatArrayType
from .enums import SVDStrategy


class KernelInputType(NamedTuple):
    """Standard data object for initializing kernels."""

    sparse_matrix_X: FloatArrayType
    low_rank_candidate_L: FloatArrayType
    target_rank: int
    svd_strategy: SVDStrategy
    tolerance: Union[float, None]


class AggressiveMomentumAdditionalParameters(NamedTuple):
    """Additional parameters for aggressive momentum model-free kernel.

    The hyperparameters determine the size of the extrapolation step (`momentum_beta`) on
    `low_rank_candidate_L` and `utility_matrix_Z` adaptively, conditional on the
    direction of change of the objective function `||X - max(0, \\Theta)||^2_F` in every
    iteration after the initial one.

    Parameters:
        - momentum_beta: Base momentum coefficient, determining the initial size
            of the extrapolation step. Default is 0.7.
        - momentum_increase_factor_gamma: Factor by which the momentum is increased
            when the objective function improves. Default is 1.1.
        - momentum_upper_bound_increase_factor_gamma_bar: Factor to increase the
            upper bound beta_bar when objective function increases. Default is 1.05.
        - momentum_decrease_divisor_eta: Divisor used to decrease momentum when
            the objective function decreases, promoting stability. Default is 2.5.
    """

    momentum_beta: float = 0.7
    momentum_increase_factor_gamma: float = 1.1
    momentum_upper_bound_increase_factor_gamma_bar: float = 1.05
    momentum_decrease_divisor_eta: float = 2.5


KernelSpecificParameters = Union[float, int, AggressiveMomentumAdditionalParameters]
