"""Public package interface for gene_prevalence."""

from .model import BernoulliPrevalenceModel
from .simulations import build_observation_matrix, build_presence_absence_matrix

__all__ = [
    "BernoulliPrevalenceModel",
    "build_observation_matrix",
    "build_presence_absence_matrix",
]
