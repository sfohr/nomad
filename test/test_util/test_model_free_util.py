import numpy as np

from fi_nomad.util.model_free_util import (
    construct_utility,
    apply_momentum,
    reconstruct_X_from_L,
)


def test_construct_utility() -> None:
    # fmt: off
    base_matrix = np.array([
        [ 1,  5,  0],
        [ 0,  0,  0]
    ])
    low_rank = np.array([
        [ 0, -1,  2],
        [-1,  1, -5]
    ])
    # Expect: non-zero values from base_matrix should be preserved
    # --> [ 1,  5,  0]
    #     [ 0,  0,  0]
    # Then negative values from low_rank should be passed to any
    # remaining 0s:
    # --> [ 1,  5,  0]
    #     [-1,  0, -5]
    expected_matrix = np.array([
        [ 1,  5,  0],
        [-1,  0, -5]
    ])
    # fmt: on
    result = construct_utility(low_rank, base_matrix)
    np.testing.assert_array_equal(expected_matrix, result)


def test_apply_momentum() -> None:
    # fmt: off
    current_matrix = np.array([
        [1, 4, 2],
        [0, 3, 0]
    ])
    previous_matrix = np.array([
        [-1, 0, 4],
        [0, 3, 1]
    ])
    momentum_parameter = 0.5

    expected_matrix = np.array([
        [2, 6, 1],
        [0, 3, -0.5]
    ])
    # fmt: on

    # momentum beta = 0.5
    result = apply_momentum(current_matrix, previous_matrix, momentum_parameter)
    np.testing.assert_array_equal(expected_matrix, result)

    # momentum beta = 1.0 doubles the step taken from previous to the current iter
    result = apply_momentum(current_matrix, previous_matrix, beta=1.0)
    np.testing.assert_array_equal(
        result - previous_matrix, 2 * (current_matrix - previous_matrix)
    )

    # momentum beta = 0.0 does not alter the matrix at all
    result = apply_momentum(current_matrix, previous_matrix, beta=0.0)
    np.testing.assert_array_equal(result, current_matrix)


def test_reconstruct_X_from_L() -> None:
    # fmt: off
    all_negative_L = np.array([
        [-1.0, -2.0],
        [-3.0, -4.0]
    ])
    all_positive_L = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    mixed_L = np.array([
        [-1.0, 2.0],
        [0.0, -4.0]
    ])
    mixed_L_expected_result = np.array([
        [0.0, 2.0],
        [0.0, 0.0]
    ])
    all_zero_L = np.zeros((2, 2))
    # fmt: on

    # all negative -> all set to zero
    result = reconstruct_X_from_L(all_negative_L)
    np.testing.assert_array_equal(result, np.zeros_like(all_negative_L))

    # all positive -> returns input
    result = reconstruct_X_from_L(all_positive_L)
    np.testing.assert_array_equal(result, all_positive_L)

    # mixed case
    result = reconstruct_X_from_L(mixed_L)
    np.testing.assert_array_equal(result, mixed_L_expected_result)

    # all zero -> returns input
    result = reconstruct_X_from_L(all_zero_L)
    np.testing.assert_array_equal(result, all_zero_L)
