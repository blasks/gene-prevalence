"""Public package interface for gene_prevalence."""

from .model import BernoulliPrevalenceModel, ConfidenceIntervalResult, PrevalenceFitResult
from .simulations import build_observed_data_matrix, build_presence_absence_matrix, build_three_part_pi_vector

__all__ = [
    "BernoulliPrevalenceModel",
    "ConfidenceIntervalResult",
    "PrevalenceFitResult",
    "build_observed_data_matrix",
    "build_presence_absence_matrix",
    "build_three_part_pi_vector",
]
