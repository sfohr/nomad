"""Defines types for objects passed to kernels in instantiation. Kernel-specific parameter
sets will also be included here.

Classes:
    KernelInputType: Standard data object for initializing kernels.
    Momentum3BlockAdditionalParameters: Additional parameters used by momentum 3-block
        model-free kernel.
    AggressiveMomentumAdditionalParameters: Additional parameters for the Aggressive
        Momentum model-free kernel.

"""

from typing import NamedTuple, Union, Optional
from .types import FloatArrayType
from .enums import SVDStrategy


class KernelInputType(NamedTuple):
    """Standard data object for initializing kernels."""

    sparse_matrix_X: FloatArrayType
    low_rank_candidate_L: FloatArrayType
    target_rank: int
    svd_strategy: SVDStrategy
    tolerance: Union[float, None]


class Momentum3BlockAdditionalParameters(NamedTuple):
    """Additional parameters for momentum 3-block model-free kernel.
    W0 and H0 are candidate low rank factors (opposed to initialization using
    low-rank candidate matrix L). beta is the momentum hyperparameter."""

    momentum_beta: float = 0.7
    candidate_factor_W0: Optional[FloatArrayType] = None
    candidate_factor_H0: Optional[FloatArrayType] = None


class AggressiveMomentumAdditionalParameters(NamedTuple):
    """Additional parameters for Aggressive Momentum model-free kernel (A-NMD).

    Additional parameters adapting the magnitude of the extrapolation step
    (`momentum_beta`) on `low_rank_candidate_L` and `utility_matrix_Z`, conditional on the
    direction of change of the loss `||X - max(0, \\Theta)||^2_F`.

    Following constraints apply:
        - :math:`\\beta \\in (0, 1)`
        - :math:`1.0 < \\gamma_bar < \\gamma < \\eta

    Parameters:
        `momentum_beta`: Initial magnitude of the extrapolation step. Defaults
            to `0.7`.
        `momentum_upper_bound_increase_factor_gamma_bar`: Factor increasing upper
            bound `beta_bar` when loss increases. Defaults to `1.05`.
        `momentum_increase_factor_gamma`: Factor increasing `momentum_beta` if loss
            decreases. Defaults to `1.1`.
        `momentum_decrease_divisor_eta`: Divisor decreasing `momentum_beta` if loss
            decreases. Defaults to `2.5`.
    """

    momentum_beta: float = 0.7
    momentum_upper_bound_increase_factor_gamma_bar: float = 1.05
    momentum_increase_factor_gamma: float = 1.1
    momentum_decrease_divisor_eta: float = 2.5


KernelSpecificParameters = Union[
    Momentum3BlockAdditionalParameters, AggressiveMomentumAdditionalParameters
]
