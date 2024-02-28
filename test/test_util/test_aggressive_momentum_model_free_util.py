from typing import cast
import numpy as np
from unittest.mock import Mock, patch
from pytest import approx
import pytest


from fi_nomad.util.aggressive_momentum_model_free_util import (
    increase_momentum_beta,
    increase_momentum_upper_bound_beta_bar,
    decrease_momentum_beta,
)


PKG = "fi_nomad.util.aggressive_momentum_model_free_util"


def test_increase_momentum_beta_regular() -> None:
    beta = 0.5
    gamma = 1.2
    beta_bar = 1.0
    expected_result = 0.6

    result = increase_momentum_beta(beta, gamma, beta_bar=beta_bar)
    assert expected_result == result


def test_increase_momentum_beta_upper_bound_reached() -> None:
    beta = 0.8
    gamma = 1.5
    beta_bar = 1.0
    expected_result = 1.0

    result = increase_momentum_beta(beta, gamma, beta_bar=beta_bar)
    assert expected_result == result


def test_increase_momentum_upper_bound_beta_bar_regular() -> None:
    beta_bar = 0.9
    gamma_bar = 0.9

    expected_result = beta_bar * gamma_bar
    actual_result = increase_momentum_upper_bound_beta_bar(beta_bar, gamma_bar)
    assert expected_result == actual_result


def test_increase_momentum_upper_bound_beta_bar_bound_reached() -> None:
    beta_bar = 1.0
    gamma_bar = 1.2

    expected_result = 1.0
    actual_result = increase_momentum_upper_bound_beta_bar(beta_bar, gamma_bar)
    assert expected_result == actual_result


def test_decrease_momentum_beta() -> None:
    beta = 0.9
    eta = 2.5

    expected_result = 0.36
    actual_result = decrease_momentum_beta(beta, eta)
    assert expected_result == actual_result
