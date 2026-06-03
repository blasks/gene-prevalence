"""Utilities for generating simulated gene family datasets."""

import numpy as np
import pandas as pd


def _coerce_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Return a NumPy random generator from a seed-like input."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _validate_pi_vector(
    n_genes: int,
    pi_vector: pd.Series | np.ndarray | list[float] | None,
) -> np.ndarray:
    """Validate or construct the gene-level prevalence vector."""
    if n_genes <= 0:
        raise ValueError("n_genes must be a positive integer.")

    if pi_vector is None:
        return np.full(n_genes, 0.5, dtype=float)

    vector = np.asarray(pi_vector, dtype=float)

    if vector.shape != (n_genes,):
        raise ValueError("pi_vector must have length n_genes.")

    if np.any((vector < 0) | (vector > 1)):
        raise ValueError("pi_vector values must fall between 0 and 1.")

    return vector


def build_presence_absence_matrix(
    n_genomes: int = 1000,
    n_genes: int = 10000,
    pi_vector: pd.Series | np.ndarray | list[float] | None = None,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """
    Build a simulated gene family presence/absence matrix.

    Parameters
    ----------
    n_genomes : int, default=1000
        Number of simulated genomes to generate.
    n_genes : int, default=10000
        Number of simulated gene families to generate.
    pi_vector : pd.Series or array-like of float or None, default=None
        Gene-level prevalence values aligned to the simulated gene columns. The
        vector must have length ``n_genes`` and values in ``[0, 1]``. If
        ``None``, a vector of length ``n_genes`` filled with ``0.5`` is used.
    rng : np.random.Generator or int or None, default=None
        Random generator or seed used to sample gene presences.

    Returns
    -------
    pd.DataFrame
        A binary matrix indexed by simulated genomes and columned by simulated
        genes, with values in ``{0, 1}``.
    """
    if n_genomes <= 0:
        raise ValueError("n_genomes must be a positive integer.")

    prevalence_vector = _validate_pi_vector(n_genes=n_genes, pi_vector=pi_vector)
    rng = _coerce_rng(rng)

    draws = rng.random((n_genomes, n_genes))
    data = (draws < prevalence_vector[np.newaxis, :]).astype(int)

    genomes = [f"sim_genome_{index}" for index in range(n_genomes)]
    genes = [f"sim_gene_{index}" for index in range(n_genes)]
    return pd.DataFrame(data, index=genomes, columns=genes)


def build_three_part_pi_vector(
    pangenome_size: int,
    genes_per_genome: int,
    n_core_genes: int,
    non_core_span: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Build a prevalence vector with a core block and two non-core segments.

    The returned vector has length ``pangenome_size``, corresponding to the
    total number of gene families in the simulated pangenome. Its first
    ``n_core_genes`` entries are set to ``1``, representing core genes found
    in every genome. The remaining entries are split around a delimiter into an
    upper segment and a lower segment so that the full vector sums to
    ``genes_per_genome``, the target number of genes per genome.

    Parameters
    ----------
    pangenome_size : int
        Total number of gene families in the simulated pangenome. This is the
        length of the returned prevalence vector.
    genes_per_genome : int
        Target number of genes per genome. This is the target sum of the
        returned prevalence vector.
    n_core_genes : int
        Number of core gene families, meaning genes with prevalence ``1`` that
        are present in every genome.
    non_core_span : tuple[float, float], default=(0.0, 1.0)
        Candidate low and high endpoints for the non-core portion of the
        prevalence vector. The two values may be given in either order and are
        adjusted relative to the delimiter as implemented below. Must be within the 
        range ``[0, 1]``.

    Returns
    -------
    np.ndarray
        The generated prevalence vector.

    Raises
    ------
    ValueError
        If the inputs are inconsistent or the adjusted span cannot produce a
        valid split.
    """
    # Basic validation
    if not (
        isinstance(pangenome_size, int)
        and isinstance(genes_per_genome, int)
        and isinstance(n_core_genes, int)
    ):
        raise ValueError(
            "pangenome_size, genes_per_genome, and n_core_genes must be integers."
        )
    if not (0 <= genes_per_genome <= pangenome_size):
        raise ValueError("Require 0 <= genes_per_genome <= pangenome_size.")
    if not (0 <= n_core_genes <= min(pangenome_size, genes_per_genome)):
        raise ValueError(
            "Require 0 <= n_core_genes <= min(pangenome_size, genes_per_genome)."
        )

    n_non_core = pangenome_size - n_core_genes
    non_core_per_genome = genes_per_genome - n_core_genes

    if n_non_core == 0:
        if non_core_per_genome != 0:
            raise ValueError(
                "With n_core_genes == pangenome_size, genes_per_genome must "
                "equal pangenome_size."
            )
        return np.ones(pangenome_size, dtype=float)

    delimiter = non_core_per_genome / n_non_core

    high_endpoint, low_endpoint = max(non_core_span), min(non_core_span)
    if not (0.0 <= high_endpoint <= 1.0 and 0.0 <= low_endpoint <= 1.0):
        raise ValueError("span values must be within [0, 1].")

    adjusted_high = high_endpoint if high_endpoint > delimiter else 1.0
    adjusted_low = low_endpoint if low_endpoint < delimiter else 0.0

    span_width = adjusted_high - adjusted_low
    if span_width <= 0:
        raise ValueError("Adjusted span invalid: need span.max() > span.min().")

    # Solve for the segment lengths before rounding upper up and lower down.
    upper_size_float = (
        2 * non_core_per_genome - (delimiter + adjusted_low) * n_non_core
    ) / span_width
    upper_size = int(np.ceil(upper_size_float))
    lower_size = n_non_core - upper_size

    if (
        upper_size < 0
        or lower_size < 0
        or upper_size > n_non_core
        or lower_size > n_non_core
    ):
        raise ValueError(
            "Infeasible sizes after rounding: "
            f"x={upper_size}, y={lower_size}, m={n_non_core}."
        )

    core = np.ones(n_core_genes, dtype=float)
    upper = (
        np.linspace(adjusted_high, delimiter, upper_size)
        if upper_size > 0
        else np.array([], dtype=float)
    )
    lower = (
        np.linspace(delimiter, adjusted_low, lower_size)
        if lower_size > 0
        else np.array([], dtype=float)
    )

    pi_vector = np.concatenate([core, upper, lower])

    return np.clip(pi_vector, 0.0, 1.0)


def build_observed_data_matrix(
    presence_absence_matrix: pd.DataFrame,
    completeness_levels: pd.Series | np.ndarray | list[float],
    rng: np.random.Generator | int | None = None,
    gene_biases: pd.Series | np.ndarray | list[float] | None = None,
    enforce_completeness: bool = False,
    return_observation_mask: bool = False, 
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mask a binary presence/absence matrix to simulate incomplete observations.

    Each genome row is assigned a completeness value between 0 and 1. A
    gene that is truly present is observed independently with probability equal
    to that genome's completeness. Absent genes remain absent.

    Parameters
    ----------
    presence_absence_matrix : pd.DataFrame
        Binary gene family presence/absence matrix with genomes in rows and
        genes in columns.
    completeness_levels : pd.Series or array-like of float
        Genome completeness values aligned to the rows of
        ``presence_absence_matrix``. Values must lie in ``[0, 1]``.
    rng : np.random.Generator or int or None, default=None
        Random generator or seed used to sample which present genes are
        observed.
    gene_biases : pd.Series or array-like of float, optional
        Gene-specific biases in observation probabilities, aligned to the columns of
        ``presence_absence_matrix``. Values must be non-negative and are interpreted 
        as multiplicative factors on the genome-level completeness values. If not
        provided, all genes are assumed to have a bias of 1.0. Values must lie in ``(0, 1]``.
    enforce_completeness : bool, default=False
        If True, enforce that the number of observed genes in each genome matches the 
        expected number given the completeness level. In this case, observation events
        are sampled from the true presence values without replacement. 
        If False, allow for stochastic variation around the expected number of observed 
        genes. In this case, observation events are modeled as independent Bernoulli 
        trials with probability equal to the completeness level, regardless of the 
        true presence values of the underlying presence/absence matrix.
    return_observation_mask : bool, default=False
        If True, return a tuple of (observation_matrix, observation_mask) where
        observation_mask is a boolean DataFrame indicating which genes were observed.
        If False, return only the observation_matrix.

    Returns
    -------
    pd.DataFrame or tuple of pd.DataFrame
        If ``return_observation_mask`` is False, returns a DataFrame representing the
        observed presence/absence matrix. If ``return_observation_mask`` is True,
        returns a tuple of (observed_data, observation_mask), where ``observation_mask``
        is a boolean DataFrame indicating which genes were observed.
    """
    if not presence_absence_matrix.map(lambda value: value in [0, 1]).all().all():
        raise ValueError("presence_absence_matrix must be binary (0 and 1).")

    if isinstance(completeness_levels, pd.Series):
        if not completeness_levels.index.equals(presence_absence_matrix.index):
            raise ValueError(
                "The index of completeness_levels must match the matrix rows."
            )
        completeness = completeness_levels.to_numpy(dtype=float)
    else:
        completeness = np.asarray(completeness_levels, dtype=float)
        if completeness.shape != (presence_absence_matrix.shape[0],):
            raise ValueError(
                "completeness_levels must have one value per matrix row."
            )
    if gene_biases is not None:
        if isinstance(gene_biases, pd.Series):
            if not gene_biases.index.equals(presence_absence_matrix.columns):
                raise ValueError(
                    "The index of gene_biases must match the matrix columns."
                )
            biases = gene_biases.to_numpy(dtype=float)
        else:
            biases = np.asarray(gene_biases, dtype=float)
            if biases.shape != (presence_absence_matrix.shape[1],):
                raise ValueError(
                    "gene_biases must have one value per matrix column."
                )
        if np.any(biases <= 0) or np.any(biases > 1):
            raise ValueError("gene_biases values must be in the range (0, 1].")
    else:
        biases = np.ones(presence_absence_matrix.shape[1], dtype=float)


    if np.any((completeness < 0) | (completeness > 1)):
        raise ValueError("completeness_levels values must fall between 0 and 1.")

    rng = _coerce_rng(rng)

    if enforce_completeness:
        observation_mask = np.zeros_like(presence_absence_matrix, dtype=int)
        for row_index, completeness_value in enumerate(completeness):
            true_presence = presence_absence_matrix.iloc[row_index, :].to_numpy(dtype=int)
            n_true_present = true_presence.sum()
            n_to_observe = int(np.round(completeness_value * n_true_present))
            if n_to_observe > 0 and n_true_present > 0:
                # norm the biases for the present genes to sum to 1 for sampling probabilities
                normed_biases = biases[true_presence == 1] / biases[true_presence == 1].sum()
                observed_indices = rng.choice(
                    np.where(true_presence == 1)[0],
                    size=n_to_observe,
                    replace=False,
                    p=normed_biases,
                )
                observation_mask[row_index, observed_indices] = 1
    else:
        # simulate observation events as independent Bernoulli trials with probability 
        # equal to the completeness level multiplied by gene-specific biases (if provided)
        adjusted_completeness = completeness[:, np.newaxis] * biases[np.newaxis, :]
        observation_mask = rng.random(presence_absence_matrix.shape) < adjusted_completeness
    
    # mask input presence/absence matrix to produce observed data
    observed_data = presence_absence_matrix.to_numpy(dtype=int) * observation_mask.astype(int)

    # convert observed data and observation mask to DataFrames
    observed_data_df = pd.DataFrame(
        observed_data,
        index=presence_absence_matrix.index,
        columns=presence_absence_matrix.columns,
    )
    observation_mask_df = pd.DataFrame(
        observation_mask.astype(bool),
        index=presence_absence_matrix.index,
        columns=presence_absence_matrix.columns,
    )

    if return_observation_mask:
        return observed_data_df, observation_mask_df
    else:
        return observed_data_df


__all__ = [
    "build_observed_data_matrix",
    "build_presence_absence_matrix",
    "build_three_part_pi_vector",
]
