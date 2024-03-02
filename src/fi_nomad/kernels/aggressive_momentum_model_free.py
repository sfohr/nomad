"""Defines the aggressive momentum model-free kernel

Classes:
    AggressiveMomentumModelFreeKernel: XXXXX.

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
)
from fi_nomad.util import find_low_rank, compute_loss, two_part_factor


class AggressiveMomentumModelFreeKernel(KernelBase):
    """Aggressive momentum model-free algorithm as described in Seraghiti et. al. (2023)"""

    def __init__(
        self,
        indata: KernelInputType,
        custom_params: AggressiveMomentumAdditionalParameters,
    ) -> None:
        super().__init__(indata)

        validate_hyperparameters(custom_params)

        self.beta_upper_bound_beta_bar = 1.0
        self.momentum_beta = custom_params.momentum_beta
        self.momentum_beta_history = [custom_params.momentum_beta]
        self.momentum_increase_factor_gamma = (
            custom_params.momentum_increase_factor_gamma
        )
        self.momentum_upper_bound_increase_factor_gamma_bar = (
            custom_params.momentum_upper_bound_increase_factor_gamma_bar
        )
        self.momentum_decrease_divisor_eta = custom_params.momentum_decrease_divisor_eta

        self.previous_low_rank_candidate_L = indata.low_rank_candidate_L.copy()
        self.utility_matrix_Z = np.empty_like(indata.low_rank_candidate_L)
        self.previous_utility_matrix_Z = indata.sparse_matrix_X.copy()

    def step(self) -> None:
        """xxxxxx"""

        if self.elapsed_iterations > 0:
            self.low_rank_candidate_L = apply_momentum(
                self.low_rank_candidate_L,
                self.previous_low_rank_candidate_L,
                self.momentum_beta,
            )
            if self.elapsed_iterations > 2:
                loss_decreasing = compute_loss(
                    self.utility_matrix_Z,
                    self.low_rank_candidate_L,
                ) < compute_loss(
                    self.previous_utility_matrix_Z,
                    self.previous_low_rank_candidate_L,
                )
                if loss_decreasing:
                    # TODO: update steps in class method?
                    self.momentum_beta = increase_momentum_beta(
                        self.momentum_beta,
                        self.momentum_increase_factor_gamma,
                        self.beta_upper_bound_beta_bar,
                    )
                    self.beta_upper_bound_beta_bar = (
                        increase_momentum_upper_bound_beta_bar(
                            self.beta_upper_bound_beta_bar,
                            self.momentum_upper_bound_increase_factor_gamma_bar,
                        )
                    )
                    self.momentum_beta_history.append(self.momentum_beta)

                    np.copyto(self.previous_utility_matrix_Z, self.utility_matrix_Z)
                    np.copyto(
                        self.previous_low_rank_candidate_L, self.low_rank_candidate_L
                    )
                else:
                    self.momentum_beta = decrease_momentum_beta(
                        self.momentum_beta, self.momentum_decrease_divisor_eta
                    )
                    self.momentum_beta_history.append(self.momentum_beta)
                    self.beta_upper_bound_beta_bar = self.momentum_beta_history[-2]

                    np.copyto(
                        self.low_rank_candidate_L, self.previous_low_rank_candidate_L
                    )

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
                utility_matrix_Z, self.low_rank_candidate_L, LossType.FROBENIUS
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
