"""Utility functions for Aggressive Momentum Model-Free Kernel from Seraghiti et. al. 2023

Functions:
    xxxx: __description__
"""


def increase_momentum_beta(beta: float, gamma: float, beta_bar: float) -> float:
    """Increase momentum parameter beta

    Args:
        beta: Current momentum parameter `beta`.
        gamma: Increase factor of `beta`.
        beta_bar: Upper bound for `beta`.

    Returns:
        The updated momentum parameter `beta`.
    """
    return min(beta_bar, beta * gamma)


def increase_momentum_upper_bound_beta_bar(beta_bar: float, gamma_bar: float) -> float:
    """Increase upper bound for momentum parameter beta.

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
