"""Utility functions for Aggressive Momentum Model-Free Kernel from Seraghiti et. al. 2023

Functions:
    increase_momentum_beta: Increase momentum parameter beta by factor gamma until 
        upper bound beta_bar.
    increase_momentum_upper_bound_beta_bar: Increase upper bound for momentum 
        parameter beta by factor gamma_bar.
    decrease_momentum_beta: Decrease momentum parameter beta by divisor eta.
    validate_hyperparameters: Validate hyperparameter constraints for the 
        aggressive momentum NMD kernel.
"""

from fi_nomad.types import AggressiveMomentumAdditionalParameters


def increase_momentum_beta(beta: float, gamma: float, beta_bar: float) -> float:
    """Increase momentum parameter beta by factor gamma until upper bound beta_bar

    Args:
        beta: Current momentum parameter `beta`.
        gamma: Increase factor of `beta`.
        beta_bar: Upper bound for `beta`.

    Returns:
        The updated momentum parameter `beta`.
    """
    return min(beta_bar, beta * gamma)


def increase_momentum_upper_bound_beta_bar(beta_bar: float, gamma_bar: float) -> float:
    """Increase upper bound for momentum parameter beta by factor gamma_bar

    Args:
        beta_bar: Current upper bound for momentum parameter beta.
        gamma_bar: Increase factor of `beta_bar`.

    Returns:
        The updated upper bound `beta_bar`.
    """
    return min(1.0, gamma_bar * beta_bar)


def decrease_momentum_beta(beta: float, eta: float) -> float:
    """Decrease momentum parameter beta by divisor eta

    Args:
        beta: Previous momentum parameter `beta`.
        eta: Divisor eta that decreases `beta`.

    Returns:
        The updated momentum parameter `beta`.
    """
    return beta / eta


def validate_hyperparameters(
    custom_params: AggressiveMomentumAdditionalParameters,
) -> None:
    """Validate hyperparameter constraints for the aggressive momentum NMD kernel

    Such that:
        - :math:`\\beta \\in (0, 1)`
        - :math:`1.0 < \\gamma_bar < \\gamma < \\eta`

    Args:
        custom_params: Object containing the kernel's custom parameters.

    Raises:
        ValueError: If any of the hyperparameters does not meet the required constraints.

    Returns:
        None
    """

    beta = custom_params.momentum_beta
    gamma = custom_params.momentum_increase_factor_gamma
    gamma_bar = custom_params.momentum_upper_bound_increase_factor_gamma_bar
    eta = custom_params.momentum_decrease_divisor_eta

    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"`momentum_beta` ({beta}) must be in the range (0, 1).")
    if not gamma_bar > 1.0:
        raise ValueError(f"`gamma_bar` ({gamma_bar}) must be greater than 1.0.")
    if not gamma_bar < gamma:
        raise ValueError(
            f"`gamma_bar` ({gamma_bar}) must be less than `gamma` ({gamma})."
        )
    if not gamma < eta:
        raise ValueError(f"`gamma` ({gamma}) must be less than `eta` ({eta}).")
