"""Defines the aggressive momentum model-free kernel (A-NMD) from Seraghiti et. al. (2023)

Classes:
    AggressiveMomentumModelFreeKernel: Model-free kernel with adaptive momentum parameter.

"""

import numpy as np
from fi_nomad.kernels.kernel_base import KernelBase
from fi_nomad.types.kernelInputTypes import KernelInputType
from fi_nomad.util.model_free_util import (
    construct_utility,
    apply_momentum,
    reconstruct_X_from_L,
)
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
)
from fi_nomad.util import find_low_rank, compute_loss, two_part_factor


class AggressiveMomentumModelFreeKernel(KernelBase):
    """Aggressive Momentum model-free algorithm as described in Seraghiti et. al. (2023)

    This is an extension of the base model-free algorithm described in Saul (2022).
    It extrapolates `Z` and `L` using momentum terms with momentum parameter `momentum_beta`
    to accelerate convergence. This momentum parameter is heuristically tuned conditional
    on the increase or decrease of the loss. If the loss:

        - decreases: increase `self.momentum_beta` by `self.momentum_increase_factor_gamma`
            until `self.beta_upper_bound_beta_bar` is reached. Accept updates on
            `self.utility_matrix_Z` and `self.low_rank_candidate_L`.

        - increases: decrease `self.momentum_beta` by `self.momentum_decrease_divisor_eta`,
            set `self.beta_upper_bound_beta_bar` to `self.last_loss_reducing_beta`,
            and reject the updates on `self.utility_matrix_Z` and `self.low_rank_candidate_L`.
    """

    def __init__(
        self,
        indata: KernelInputType,
        custom_params: AggressiveMomentumAdditionalParameters,
    ) -> None:
        super().__init__(indata)

        self.parameter_update_loss = float("inf")

        # Hyperparameters
        validate_hyperparameters(custom_params)

        self.beta_upper_bound_beta_bar = 1.0
        self.momentum_beta = custom_params.momentum_beta
        self.last_loss_reducing_beta = custom_params.momentum_beta
        self.momentum_increase_factor_gamma = (
            custom_params.momentum_increase_factor_gamma
        )
        self.momentum_upper_bound_increase_factor_gamma_bar = (
            custom_params.momentum_upper_bound_increase_factor_gamma_bar
        )
        self.momentum_decrease_divisor_eta = custom_params.momentum_decrease_divisor_eta

        # Initial matrices
        self.previous_low_rank_candidate_L = indata.low_rank_candidate_L.copy()
        self.previous_utility_matrix_Z = indata.sparse_matrix_X.copy()
        self.utility_matrix_Z = indata.sparse_matrix_X.copy()

    def increase_momentum_parameters(self) -> None:
        """Increase momentum beta and update beta's upper bound beta_bar

        If the loss between previous and current step decreased, `self.momentum_beta`
        is increased by `self.momentum_upper_bound_increase_factor_gamma_bar`
        until it reaches `self.beta_upper_bound_beta_bar`.
        `self.beta_upper_bound_beta_bar` is increased by `self.momentum_increase_factor_gamma`.
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

        If the loss between previous and current step increased, `self.momentum_beta`
        is decreased by `self.momentum_decrease_divisor_eta`, thereby reducing the
        magnitude of the momentum terms. Furthermore, `self.beta_upper_bound_beta_bar`
        is set to the last beta that decreased the loss.

        Returns:
            None
        """
        last_beta_that_decreased_loss = self.momentum_beta
        self.momentum_beta = decrease_momentum_beta(
            self.momentum_beta, self.momentum_decrease_divisor_eta
        )

        self.beta_upper_bound_beta_bar = last_beta_that_decreased_loss

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

    def compute_parameter_update_loss(self) -> float:
        """Compute parameter update loss after momentum step on `low_rank_candidate_L`

        Computes the Frobenius norm of the difference between the post-momentum
        sparse reconstruction `numpy.maximum(0.0, low_rank_candidate_L)` and
        `self.sparse_matrix_X`. Used to guide momentum adaption.

        Returns:
            None
        """
        parameter_update_loss = compute_loss(
            self.sparse_matrix_X,
            reconstruct_X_from_L(self.low_rank_candidate_L),
            LossType.FROBENIUS,
        )
        return parameter_update_loss

    def step(self) -> None:
        """Executes a single step of the aggressive momentum model-free algorithm (A-NMD).

        A single step of A-NMD consists of:

            - Applies momentum to the low-rank candidate `L` (starting from the 2nd iteration).
            - Adjusts the momentum parameter beta (starting from the 3rd iteration).
            - Constructs the utility matrix `Z`.
            - Applies momentum to the utility matrix `Z`.
            - Updates the low-rank candidate `L`.
            - Computes the loss (if a tolerance is specified).

        This implementation differs from the matlab implementation and the algorithm described
        in the paper in the order of operations. Momentum on `low_rank_condidate_L` is
        performed at the beginning of each iteration, starting in iteration 1, so we can omit
        an additional parameter `max_iterations` and ensure that `self.low_rank_candidate_L` is
        the matrix that produced the current `self.loss`. Furthermore, the momentum
        parameter updates are moved to the top for the same reason.

        Returns:
            None
        """
        if self.elapsed_iterations > 0:
            self.low_rank_candidate_L = apply_momentum(
                self.low_rank_candidate_L,
                self.previous_low_rank_candidate_L,
                self.momentum_beta,
            )

            previous_parameter_update_loss = self.parameter_update_loss
            self.parameter_update_loss = self.compute_parameter_update_loss()

            if self.elapsed_iterations > 2:
                if self.parameter_update_loss < previous_parameter_update_loss:
                    self.last_loss_reducing_beta = self.momentum_beta
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
