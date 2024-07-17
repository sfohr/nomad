from typing import Tuple, cast
import numpy as np
from unittest.mock import Mock, patch

from pytest import fixture
from fi_nomad.kernels import AggressiveMomentumModelFreeKernel
from fi_nomad.types import (
    FloatArrayType,
    KernelInputType,
    AggressiveMomentumAdditionalParameters,
    LossType,
    SVDStrategy,
)

Fixture = Tuple[
    KernelInputType,
    AggressiveMomentumAdditionalParameters,
    AggressiveMomentumModelFreeKernel,
]
PKG = "fi_nomad.kernels.aggressive_momentum_model_free"
KERNEL_CLASS = "AggressiveMomentumModelFreeKernel"


@fixture
def fixture_with_tol() -> Fixture:
    sparse = np.eye(9) * 3.0
    (m, n) = sparse.shape
    target_rank = 5
    candidate_W = np.ones((m, target_rank))
    candidate_H = np.ones((target_rank, n))
    candidate = candidate_W @ candidate_H
    svd_strategy = SVDStrategy.RANDOM_TRUNCATED
    tolerance = 3.0
    indata = KernelInputType(sparse, candidate, target_rank, svd_strategy, tolerance)
    kernel_params = AggressiveMomentumAdditionalParameters(
        momentum_beta=0.7,
        momentum_increase_factor_gamma=1.1,
        momentum_upper_bound_increase_factor_gamma_bar=1.05,
        momentum_decrease_divisor_eta=2.5,
    )
    kernel = AggressiveMomentumModelFreeKernel(indata, kernel_params)
    return (indata, kernel_params, kernel)


def test_aggressive_momentum_kernel_instantiation(fixture_with_tol: Fixture) -> None:
    (indata, kernel_params, kernel) = fixture_with_tol
    # hyperparameters
    assert kernel.momentum_beta == kernel_params.momentum_beta
    assert kernel.last_loss_reducing_beta == kernel_params.momentum_beta
    assert kernel.parameter_update_loss == float("inf")
    assert (
        kernel.momentum_upper_bound_increase_factor_gamma_bar
        == kernel_params.momentum_upper_bound_increase_factor_gamma_bar
    )
    assert (
        kernel.momentum_increase_factor_gamma
        == kernel_params.momentum_increase_factor_gamma
    )
    assert (
        kernel.momentum_decrease_divisor_eta
        == kernel_params.momentum_decrease_divisor_eta
    )

    # matrices
    np.testing.assert_array_equal(
        kernel.low_rank_candidate_L, indata.low_rank_candidate_L
    )
    np.testing.assert_array_equal(
        kernel.previous_low_rank_candidate_L, indata.low_rank_candidate_L
    )
    np.testing.assert_array_equal(kernel.utility_matrix_Z, indata.sparse_matrix_X)
    np.testing.assert_array_equal(
        kernel.previous_utility_matrix_Z, indata.sparse_matrix_X
    )


@patch(f"{PKG}.compute_loss")
@patch(f"{PKG}.find_low_rank")
@patch(f"{PKG}.apply_momentum")
@patch(f"{PKG}.construct_utility")
@patch(f"{PKG}.{KERNEL_CLASS}.reject_matrix_updates")
@patch(f"{PKG}.{KERNEL_CLASS}.decrease_momentum_parameters")
@patch(f"{PKG}.{KERNEL_CLASS}.accept_matrix_updates")
@patch(f"{PKG}.{KERNEL_CLASS}.increase_momentum_parameters")
@patch(f"{PKG}.{KERNEL_CLASS}.compute_parameter_update_loss")
def test_aggressive_momentum_first_kernel_step(
    mock_compute_parameter_update_loss: Mock,
    mock_increase_momentum_parameters: Mock,
    mock_accept_matrix_updates: Mock,
    mock_decrease_momentum_parameters: Mock,
    mock_reject_matrix_updates: Mock,
    mock_construct: Mock,
    mock_apply_momentum: Mock,
    mock_find_low_rank: Mock,
    mock_compute_loss: Mock,
    fixture_with_tol: Fixture,
) -> None:
    (indata, _, kernel) = fixture_with_tol
    shape_X = indata.sparse_matrix_X.shape
    mock_construct.return_value = np.full(shape_X, 4.0)
    mock_apply_momentum.return_value = np.full(shape_X, 5.0)
    mock_find_low_rank.return_value = np.full(shape_X, 6.0)
    mock_compute_loss.return_value = 3.0

    initial_utility_Z = kernel.utility_matrix_Z

    kernel.step()
    mock_construct.assert_called_once_with(
        indata.low_rank_candidate_L, indata.sparse_matrix_X
    )
    mock_apply_momentum.assert_called_once_with(
        mock_construct.return_value,
        initial_utility_Z,
        kernel.momentum_beta,
    )
    mock_find_low_rank.assert_called_once_with(
        mock_apply_momentum.return_value,
        kernel.target_rank,
        indata.low_rank_candidate_L,
        kernel.svd_strategy,
    )

    mock_compute_loss.assert_called_once_with(
        mock_apply_momentum.return_value,
        mock_find_low_rank.return_value,
        LossType.FROBENIUS,
    )

    mock_compute_parameter_update_loss.assert_not_called()
    mock_increase_momentum_parameters.assert_not_called()
    mock_accept_matrix_updates.assert_not_called()
    mock_decrease_momentum_parameters.assert_not_called()
    mock_reject_matrix_updates.assert_not_called()


@patch(f"{PKG}.compute_loss")
@patch(f"{PKG}.reconstruct_X_from_L")
def test_compute_parameter_update_loss(
    mock_reconstruct_X_from_L: Mock,
    mock_compute_loss: Mock,
    fixture_with_tol: Fixture,
) -> None:
    (_, _, kernel) = fixture_with_tol
    kernel.parameter_update_loss = 1.0
    kernel.elapsed_iterations = 1

    mock_reconstruct_X_from_L.return_value = np.eye(3)

    kernel.compute_parameter_update_loss()
    mock_reconstruct_X_from_L.assert_called_once_with(kernel.low_rank_candidate_L)
    mock_compute_loss.assert_called_once_with(
        kernel.sparse_matrix_X,
        mock_reconstruct_X_from_L.return_value,
        LossType.FROBENIUS,
    )


@patch(f"{PKG}.{KERNEL_CLASS}.compute_parameter_update_loss")
def test_compute_parameter_update_loss_correct_assignment(
    mock_compute_parameter_update_loss: Mock,
    fixture_with_tol: Fixture,
) -> None:
    (_, _, kernel) = fixture_with_tol
    kernel.parameter_update_loss = 1.0
    kernel.elapsed_iterations = 1
    mock_compute_parameter_update_loss.return_value = 3.0

    kernel.step()
    mock_compute_parameter_update_loss.assert_called_once()
    assert kernel.parameter_update_loss == 3.0


@patch(f"{PKG}.increase_momentum_beta")
@patch(f"{PKG}.increase_momentum_upper_bound_beta_bar")
def test_aggressive_momentum_increase_momentum_parameters(
    mock_increase_momentum_upper_bound_beta_bar: Mock,
    mock_increase_momentum_beta: Mock,
    fixture_with_tol: Fixture,
) -> None:
    (_, _, kernel) = fixture_with_tol

    beta_new_val = 0.999
    beta_bar_new_val = 1.222

    mock_increase_momentum_beta.return_value = beta_new_val
    mock_increase_momentum_upper_bound_beta_bar.return_value = beta_bar_new_val

    kernel.increase_momentum_parameters()
    assert kernel.momentum_beta == beta_new_val
    assert kernel.beta_upper_bound_beta_bar == beta_bar_new_val


@patch(f"{PKG}.decrease_momentum_beta")
def test_aggressive_momentum_decrease_momentum_parameters(
    mock_decrease_momentum_beta: Mock,
    fixture_with_tol: Fixture,
) -> None:
    (_, kernel_params, kernel) = fixture_with_tol

    beta_new_val = 0.111
    beta_bar_new_val = kernel_params.momentum_beta

    mock_decrease_momentum_beta.return_value = beta_new_val

    kernel.decrease_momentum_parameters()
    assert kernel.momentum_beta == beta_new_val
    assert kernel.beta_upper_bound_beta_bar == beta_bar_new_val


def test_aggressive_momentum_accept_matrix_updates(fixture_with_tol: Fixture) -> None:
    (_, _, kernel) = fixture_with_tol

    previous_utility_matrix_Z = kernel.previous_utility_matrix_Z.copy()
    utility_matrix_Z = previous_utility_matrix_Z + 1.0

    previous_low_rank_candidate_L = kernel.previous_low_rank_candidate_L.copy()
    low_rank_candidate_L = previous_low_rank_candidate_L * 1.1

    kernel.low_rank_candidate_L = low_rank_candidate_L
    kernel.utility_matrix_Z = utility_matrix_Z

    kernel.accept_matrix_updates()
    np.testing.assert_array_equal(
        kernel.previous_low_rank_candidate_L, low_rank_candidate_L
    )
    np.testing.assert_array_equal(kernel.previous_utility_matrix_Z, utility_matrix_Z)


def test_aggressive_momentum_reject_matrix_updates(fixture_with_tol: Fixture) -> None:
    (_, _, kernel) = fixture_with_tol

    previous_utility_matrix_Z = kernel.previous_utility_matrix_Z.copy()
    utility_matrix_Z = previous_utility_matrix_Z + 1.0

    previous_low_rank_candidate_L = kernel.previous_low_rank_candidate_L.copy()
    low_rank_candidate_L = previous_low_rank_candidate_L * 1.1

    kernel.low_rank_candidate_L = low_rank_candidate_L
    kernel.utility_matrix_Z = utility_matrix_Z

    kernel.reject_matrix_updates()
    np.testing.assert_array_equal(
        kernel.low_rank_candidate_L, previous_low_rank_candidate_L
    )
    np.testing.assert_array_equal(kernel.utility_matrix_Z, previous_utility_matrix_Z)


@patch(f"{PKG}.{KERNEL_CLASS}.reject_matrix_updates")
@patch(f"{PKG}.{KERNEL_CLASS}.decrease_momentum_parameters")
@patch(f"{PKG}.{KERNEL_CLASS}.accept_matrix_updates")
@patch(f"{PKG}.{KERNEL_CLASS}.increase_momentum_parameters")
@patch(f"{PKG}.{KERNEL_CLASS}.compute_parameter_update_loss")
def test_aggressive_momentum_parameter_adaption(
    mock_compute_parameter_update_loss: Mock,
    mock_increase_momentum_parameters: Mock,
    mock_accept_matrix_updates: Mock,
    mock_decrease_momentum_parameters: Mock,
    mock_reject_matrix_updates: Mock,
    fixture_with_tol: Fixture,
) -> None:
    (_, _, kernel) = fixture_with_tol
    kernel.elapsed_iterations = 3

    # loss decreasing
    kernel.parameter_update_loss = 5.0
    mock_compute_parameter_update_loss.return_value = 4.0
    kernel.step()

    mock_compute_parameter_update_loss.assert_called_once()
    mock_increase_momentum_parameters.assert_called_once()
    mock_accept_matrix_updates.assert_called_once()
    mock_decrease_momentum_parameters.assert_not_called()
    mock_reject_matrix_updates.assert_not_called()

    mock_compute_parameter_update_loss.reset_mock()
    mock_increase_momentum_parameters.reset_mock()
    mock_accept_matrix_updates.reset_mock()

    # loss increasing
    mock_compute_parameter_update_loss.return_value = 5.0
    kernel.step()

    mock_compute_parameter_update_loss.assert_called_once()
    mock_increase_momentum_parameters.assert_not_called()
    mock_accept_matrix_updates.assert_not_called()
    mock_decrease_momentum_parameters.assert_called_once()
    mock_reject_matrix_updates.assert_called_once()

    mock_compute_parameter_update_loss.reset_mock()
    mock_decrease_momentum_parameters.reset_mock()
    mock_reject_matrix_updates.reset_mock()

    # loss constant
    mock_compute_parameter_update_loss.return_value = 5.0
    kernel.step()
    mock_compute_parameter_update_loss.assert_called_once()
    mock_compute_parameter_update_loss.ass
    mock_increase_momentum_parameters.assert_not_called()
    mock_accept_matrix_updates.assert_not_called()
    mock_decrease_momentum_parameters.assert_called_once()
    mock_reject_matrix_updates.assert_called_once()


def test_aggressive_momentum_running_report(fixture_with_tol: Fixture) -> None:
    (_, _, kernel) = fixture_with_tol
    first_txt = kernel.running_report()
    assert "iteration" in first_txt
    assert "loss" in first_txt
    kernel.tolerance = None
    second_txt = kernel.running_report()
    assert second_txt == ""


def test_aggressive_momentum_final_report(fixture_with_tol: Fixture) -> None:
    (indata, _, kernel) = fixture_with_tol

    kernel.tolerance = None

    result_1 = kernel.report()
    assert "Not Tracked" in result_1.summary
    np.testing.assert_allclose(
        indata.low_rank_candidate_L,
        cast(
            FloatArrayType,
            result_1.data.factors[0] @ result_1.data.factors[1],
        ),
    )

    kernel.loss = 3.0
    result_2 = kernel.report()
    assert "3.0" in result_2.summary
