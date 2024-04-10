"""Defines the aggressive momentum model-free kernel from Seraghiti et. al. (2023)

Classes:
    AggressiveMomentumModelFreeKernel: Model-free kernel with Nesterov-type extrapolation.

"""

import numpy as np
from fi_nomad.kernels.kernel_base import KernelBase
from fi_nomad.types.kernelInputTypes import KernelInputType
from fi_nomad.util.model_free_util import construct_utility, apply_momentum
from fi_nomad.util.aggressive_momentum_model_free_util import (
    increase_momentum_beta,
    increase_momentum_upper_bound_beta_bar,
    decrease_momentum_beta,
    validate_hyperparameters,
)

from fi_nomad.types import (
    AggressiveMomentumAdditionalParameters,
    AggressiveMomentumModelFreeKernelReturnType,
    KernelReturnType,
    LossType,
    FloatArrayType,
)
from fi_nomad.util import find_low_rank, compute_loss, two_part_factor


class AggressiveMomentumModelFreeKernel(KernelBase):
    """Aggressive momentum model-free algorithm as described in Seraghiti et. al. (2023)

    The algorithm constructs a utility matrix Z that enforces the
    non-zero values of the sparse matrix, extrapolates Z with a momentum term
    and uses SVD on Z to create a candidate low-rank matrix L which is also
    extrapolated (in the subsequent step to not alter the rank).
    The size of the extrapolation steps on Z and L is determined by
    hyperparameter `momentum_beta` which is heuristically adapted conditional
    on the loss.

    If loss is decreasing:
        - increase `momentum_beta`by factor `momentum_increase_factor_gamma` until it
        reaches `beta_upper_bound_beta_bar`.
        - accept updates on Z and L.
    if loss is increasing:
        - decrease `momentum_beta` by `momentum_decrease_divisor_eta`.
        - assign last value of `momentum_beta` that decreased the loss as the new
        `beta_upper_bound_beta_bar`.
        - reject updates on Z and L.

    Note: This implementation diverges from Seraghiti et al.'s original Matlab
        implementation in terms of specific operations order and the handling of the
        extrapolation step on L to avoid additional parameters for iteration control.

    For `momentum_beta = 0`, the algorithm simplifies to the base model-free
    algorithm.
    """

    def __init__(
        self,
        indata: KernelInputType,
        custom_params: AggressiveMomentumAdditionalParameters,
    ) -> None:
        super().__init__(indata)

        # Hyperparameters
        validate_hyperparameters(custom_params)

        self.beta_upper_bound_beta_bar: float = 1.0
        self.momentum_beta: float = custom_params.momentum_beta
        self.momentum_increase_factor_gamma: float = (
            custom_params.momentum_increase_factor_gamma
        )
        self.momentum_upper_bound_increase_factor_gamma_bar: float = (
            custom_params.momentum_upper_bound_increase_factor_gamma_bar
        )
        self.momentum_decrease_divisor_eta: float = (
            custom_params.momentum_decrease_divisor_eta
        )

        # Initial matrices
        self.previous_low_rank_candidate_L: FloatArrayType = (
            indata.low_rank_candidate_L.copy()
        )
        self.previous_utility_matrix_Z: FloatArrayType = indata.sparse_matrix_X.copy()
        self.utility_matrix_Z: FloatArrayType = indata.sparse_matrix_X.copy()

    def increase_momentum_parameters(self) -> None:
        """Increase momentum beta and update beta's upper bound beta_bar

        If the loss between previous and current step decreased, the momentum
        parameter beta is increased by factor gamma until it reaches it's upper
        bound beta_bar. beta's upper bound beta_bar is increased by factor gamma_bar.
        A method call increases the magnitude of the momentum steps performed on
        utility matrix Z and low rank candidate L.

        Returns:
            None
        """
        self.momentum_beta = increase_momentum_beta(
            self.momentum_beta,
            self.momentum_increase_factor_gamma,
            self.beta_upper_bound_beta_bar,
        )
        self.beta_upper_bound_beta_bar = increase_momentum_upper_bound_beta_bar(
            self.beta_upper_bound_beta_bar,
            self.momentum_upper_bound_increase_factor_gamma_bar,
        )

    def decrease_momentum_parameters(self) -> None:
        """Decrease momentum beta by divisor eta and assign previous beta to beta_bar

        If the loss between previous and current step increased, the momentum
        parameter beta is decreased by dividing it by eta, thereby reducing the
        magnitude of the momentum terms. Furthermore, the upper bound for beta
        (beta_bar) is set to the last beta that decreased the loss.

        Returns:
            None
        """
        last_beta_that_decreased_error = self.momentum_beta
        self.momentum_beta = decrease_momentum_beta(
            self.momentum_beta, self.momentum_decrease_divisor_eta
        )

        self.beta_upper_bound_beta_bar = last_beta_that_decreased_error

    def accept_matrix_updates(self) -> None:
        """Accept updates on utility matrix Z and low rank candidate L

        Accepts updates by copying the current utility matrix and low rank
        candidate matrix to their previous versions.

        Returns:
            None
        """
        np.copyto(self.previous_utility_matrix_Z, self.utility_matrix_Z)
        np.copyto(self.previous_low_rank_candidate_L, self.low_rank_candidate_L)

    def reject_matrix_updates(self) -> None:
        """Reject updates on utility matrix Z and low rank candidate L

        Rejects updates by copying the previous low rank candidate L to the
        current low rank candidate L and the previous utility matrix Z to current.

        Returns:
            None
        """
        np.copyto(self.utility_matrix_Z, self.previous_utility_matrix_Z)
        np.copyto(self.low_rank_candidate_L, self.previous_low_rank_candidate_L)

    def loss_is_decreasing(self) -> bool:
        """Checks if loss between previous and current step decreased

        Computes the Frobenius norm between utility matrix Z and low rank candidate L
        for the current and the previous iteration and checks if the current value
        is lower than than the previous.

        Returns:
            `True`if decreasing, else `False`.
        """
        current_loss = compute_loss(
            self.utility_matrix_Z, self.low_rank_candidate_L, LossType.FROBENIUS
        )
        previous_loss = compute_loss(
            self.previous_utility_matrix_Z,
            self.previous_low_rank_candidate_L,
            LossType.FROBENIUS,
        )
        return current_loss < previous_loss

    def step(self) -> None:
        """Performs a single step of the aggressive momentum model-free algorithm.


        It first applies momentum to the low-rank candidate L if the elapsed
        iterations is greater than 0. Then if elapsed iteration is greater than 2,
        it checks if the loss is decreasing and adjusts the momentum parameters accordingly.
        Finally, it constructs utility matrix Z, applies momentum to it, and
        finds the low-rank candidate L.

        Returns:
            None
        """
        if self.elapsed_iterations > 0:
            self.low_rank_candidate_L = apply_momentum(
                self.low_rank_candidate_L,
                self.previous_low_rank_candidate_L,
                self.momentum_beta,
            )
            if self.elapsed_iterations > 2:
                if self.loss_is_decreasing():
                    self.increase_momentum_parameters()
                    self.accept_matrix_updates()
                else:
                    self.decrease_momentum_parameters()
                    self.reject_matrix_updates()

        utility_matrix_Z = construct_utility(
            self.low_rank_candidate_L, self.sparse_matrix_X
        )

        self.utility_matrix_Z = apply_momentum(
            utility_matrix_Z, self.utility_matrix_Z, self.momentum_beta
        )

        self.low_rank_candidate_L = find_low_rank(
            self.utility_matrix_Z,
            self.target_rank,
            self.low_rank_candidate_L,
            self.svd_strategy,
        )

        if self.tolerance is not None:
            self.loss = compute_loss(
                self.utility_matrix_Z, self.low_rank_candidate_L, LossType.FROBENIUS
            )

    def running_report(self) -> str:
        """Reports the current iteration number and loss. Only operative if a
        tolerance was set.

        Returns:
            Description of current iteration and loss.
        """
        txt = (
            ""
            if self.tolerance is None
            else f"iteration: {self.elapsed_iterations} loss: {self.loss}"
        )
        return txt

    def report(self) -> KernelReturnType:
        """Returns final low-rank approximation and descriptive string.
        The description indicates the total number of iterations completed,
        and the final loss (if a tolerance was set).

        Returns:
            Object containing results and summary
        """
        floss = str(self.loss) if self.loss != float("inf") else "Not Tracked"
        text = f"{self.elapsed_iterations} total, final loss {floss}"
        data = AggressiveMomentumModelFreeKernelReturnType(
            two_part_factor(self.low_rank_candidate_L)
        )
        return KernelReturnType(text, data)
