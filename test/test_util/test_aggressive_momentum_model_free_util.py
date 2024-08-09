import pytest


from fi_nomad.util.aggressive_momentum_model_free_util import (
    increase_momentum_beta,
    increase_momentum_upper_bound_beta_bar,
    decrease_momentum_beta,
    validate_hyperparameters,
)

from fi_nomad.types.kernelInputTypes import AggressiveMomentumAdditionalParameters


PKG = "fi_nomad.util.aggressive_momentum_model_free_util"


def test_increase_momentum_beta_regular() -> None:
    beta = 0.5
    gamma = 1.2
    beta_bar = 1.0
    expected_result = min(beta_bar, beta * gamma)

    result = increase_momentum_beta(beta, gamma, beta_bar=beta_bar)
    assert expected_result == result


def test_increase_momentum_beta_upper_bound_reached() -> None:
    beta = 0.8
    gamma = 1.5
    beta_bar = 1.0
    expected_result = min(beta_bar, beta * gamma)

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

    expected_result = beta / eta
    actual_result = decrease_momentum_beta(beta, eta)
    assert expected_result == actual_result


def test_validate_hyperparameters_valid_params() -> None:
    params = AggressiveMomentumAdditionalParameters(
        momentum_beta=0.5,
        momentum_increase_factor_gamma=1.2,
        momentum_upper_bound_increase_factor_gamma_bar=1.1,
        momentum_decrease_divisor_eta=1.5,
    )
    validate_hyperparameters(params)  # No exception should be raised


def test_validate_hyperparameters_invalid_momentum_beta_too_low() -> None:
    params = AggressiveMomentumAdditionalParameters(
        momentum_beta=-0.1,
        momentum_increase_factor_gamma=1.2,
        momentum_upper_bound_increase_factor_gamma_bar=1.1,
        momentum_decrease_divisor_eta=1.5,
    )
    with pytest.raises(ValueError) as excinfo:
        validate_hyperparameters(params)
    assert str(excinfo.value) == "`momentum_beta` (-0.1) must be in the range (0, 1)."


def test_validate_hyperparameters_invalid_momentum_beta_too_high() -> None:
    params = AggressiveMomentumAdditionalParameters(
        momentum_beta=1.5,
        momentum_increase_factor_gamma=1.2,
        momentum_upper_bound_increase_factor_gamma_bar=1.1,
        momentum_decrease_divisor_eta=1.5,
    )
    with pytest.raises(ValueError) as excinfo:
        validate_hyperparameters(params)
    assert str(excinfo.value) == "`momentum_beta` (1.5) must be in the range (0, 1)."


def test_validate_hyperparameters_invalid_gamma_bar_too_low() -> None:
    params = AggressiveMomentumAdditionalParameters(
        momentum_beta=0.5,
        momentum_increase_factor_gamma=1.2,
        momentum_upper_bound_increase_factor_gamma_bar=1.0,
        momentum_decrease_divisor_eta=1.5,
    )
    with pytest.raises(ValueError) as excinfo:
        validate_hyperparameters(params)
    assert str(excinfo.value) == "`gamma_bar` (1.0) must be greater than 1.0."


def test_validate_hyperparameters_invalid_gamma_bar_equal_to_gamma() -> None:
    params = AggressiveMomentumAdditionalParameters(
        momentum_beta=0.5,
        momentum_increase_factor_gamma=1.2,
        momentum_upper_bound_increase_factor_gamma_bar=1.2,
        momentum_decrease_divisor_eta=1.5,
    )
    with pytest.raises(ValueError) as excinfo:
        validate_hyperparameters(params)
    assert str(excinfo.value) == "`gamma_bar` (1.2) must be less than `gamma` (1.2)."


def test_validate_hyperparameters_invalid_gamma_bar_higher_than_gamma() -> None:
    params = AggressiveMomentumAdditionalParameters(
        momentum_beta=0.5,
        momentum_increase_factor_gamma=1.2,
        momentum_upper_bound_increase_factor_gamma_bar=1.5,
        momentum_decrease_divisor_eta=1.5,
    )
    with pytest.raises(ValueError) as excinfo:
        validate_hyperparameters(params)
    assert str(excinfo.value) == "`gamma_bar` (1.5) must be less than `gamma` (1.2)."


def test_validate_hyperparameters_invalid_gamma_equal_to_eta() -> None:
    params = AggressiveMomentumAdditionalParameters(
        momentum_beta=0.5,
        momentum_increase_factor_gamma=1.2,
        momentum_upper_bound_increase_factor_gamma_bar=1.1,
        momentum_decrease_divisor_eta=1.0,
    )
    with pytest.raises(ValueError) as excinfo:
        validate_hyperparameters(params)
    assert str(excinfo.value) == "`gamma` (1.2) must be less than `eta` (1.0)."


def test_validate_hyperparameters_invalid_gamma_higher_than_eta() -> None:
    params = AggressiveMomentumAdditionalParameters(
        momentum_beta=0.5,
        momentum_increase_factor_gamma=1.2,
        momentum_upper_bound_increase_factor_gamma_bar=1.1,
        momentum_decrease_divisor_eta=1.0,
    )
    with pytest.raises(ValueError) as excinfo:
        validate_hyperparameters(params)
    assert str(excinfo.value) == "`gamma` (1.2) must be less than `eta` (1.0)."
