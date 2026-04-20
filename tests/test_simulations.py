import pandas as pd
import pytest

from gene_prevalence.simulations import (
    build_observation_matrix,
    build_presence_absence_matrix,
)


def test_build_presence_absence_matrix_returns_expected_shape_and_labels():
    result = build_presence_absence_matrix(
        n_genomes=3,
        n_genes=5,
        pi_vector=[1.0, 0.8, 0.5, 0.2, 0.0],
        rng=0,
    )

    assert result.shape == (5, 3)
    assert list(result.index) == [f"sim_gene_{i}" for i in range(5)]
    assert list(result.columns) == [f"sim_genome_{i}" for i in range(3)]
    assert set(result.to_numpy().ravel()) <= {0, 1}


def test_build_presence_absence_matrix_uses_supplied_pi_vector():
    result = build_presence_absence_matrix(
        n_genomes=4,
        n_genes=3,
        pi_vector=[1.0, 0.0, 1.0],
        rng=1,
    )

    assert result.iloc[0].eq(1).all()
    assert result.iloc[1].eq(0).all()
    assert result.iloc[2].eq(1).all()


def test_build_presence_absence_matrix_defaults_to_half_prevalence():
    result = build_presence_absence_matrix(n_genomes=4, n_genes=3, rng=0)

    expected = pd.DataFrame(
        {
            "sim_genome_0": [0, 0, 0],
            "sim_genome_1": [1, 0, 0],
            "sim_genome_2": [1, 0, 0],
            "sim_genome_3": [1, 0, 1],
        },
        index=[f"sim_gene_{i}" for i in range(3)],
    )
    pd.testing.assert_frame_equal(result, expected)


def test_build_presence_absence_matrix_rejects_invalid_pi_vector():
    with pytest.raises(ValueError, match="must have length n_genes"):
        build_presence_absence_matrix(n_genomes=2, n_genes=3, pi_vector=[0.1, 0.2])

    with pytest.raises(ValueError, match="fall between 0 and 1"):
        build_presence_absence_matrix(
            n_genomes=2,
            n_genes=3,
            pi_vector=[0.1, 1.2, 0.4],
        )


def test_build_observation_matrix_masks_present_genes_using_completeness_series():
    data = pd.DataFrame(
        {
            "genome_1": [1, 1, 0],
            "genome_2": [1, 0, 1],
        },
        index=["gene_1", "gene_2", "gene_3"],
    )
    completeness = pd.Series([1.0, 0.0], index=data.columns)

    result = build_observation_matrix(data, completeness, rng=0)

    expected = pd.DataFrame(
        {
            "genome_1": [1, 1, 0],
            "genome_2": [0, 0, 0],
        },
        index=data.index,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_build_observation_matrix_accepts_array_like_completeness_levels():
    data = pd.DataFrame(
        {
            "genome_1": [1, 0],
            "genome_2": [1, 1],
            "genome_3": [0, 1],
        },
        index=["gene_1", "gene_2"],
    )

    result = build_observation_matrix(data, [1.0, 0.0, 1.0], rng=0)

    expected = pd.DataFrame(
        {
            "genome_1": [1, 0],
            "genome_2": [0, 0],
            "genome_3": [0, 1],
        },
        index=data.index,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_build_observation_matrix_rejects_invalid_inputs():
    data = pd.DataFrame({"genome_1": [1, 2]}, index=["gene_1", "gene_2"])

    with pytest.raises(ValueError, match="must be binary"):
        build_observation_matrix(data, [1.0], rng=0)

    valid_data = pd.DataFrame({"genome_1": [1], "genome_2": [0]}, index=["gene_1"])

    with pytest.raises(ValueError, match="must match the matrix columns"):
        build_observation_matrix(
            valid_data,
            pd.Series([1.0, 0.5], index=["genome_1", "genome_x"]),
            rng=0,
        )

    with pytest.raises(ValueError, match="one value per matrix column"):
        build_observation_matrix(valid_data, [1.0], rng=0)

    with pytest.raises(ValueError, match="fall between 0 and 1"):
        build_observation_matrix(valid_data, [1.0, 1.2], rng=0)
