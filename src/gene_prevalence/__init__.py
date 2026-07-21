"""Public package interface for gene_prevalence."""

from .model import (
    BernoulliPrevalenceModel,
    CombinedPrevalenceFitResult,
    PrevalenceFitResult,
    combine_approx_var_likelihood,
)
from .simulations import build_observed_data_matrix, build_presence_absence_matrix, build_three_part_pi_vector

__all__ = [
    "BernoulliPrevalenceModel",
    "CombinedPrevalenceFitResult",
    "PrevalenceFitResult",
    "combine_approx_var_likelihood",
    "build_observed_data_matrix",
    "build_presence_absence_matrix",
    "build_three_part_pi_vector",
]
