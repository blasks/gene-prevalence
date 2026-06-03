import pandas as pd
import pytest

from gene_prevalence.simulations import (
    build_observed_data_matrix,
    build_presence_absence_matrix,
)


def test_build_presence_absence_matrix_returns_expected_shape_and_labels():
    result = build_presence_absence_matrix(
        n_genomes=3,
        n_genes=5,
        pi_vector=[1.0, 0.8, 0.5, 0.2, 0.0],
        rng=0,
    )

    assert result.shape == (3, 5)
    assert list(result.index) == [f"sim_genome_{i}" for i in range(3)]
    assert list(result.columns) == [f"sim_gene_{i}" for i in range(5)]
    assert set(result.to_numpy().ravel()) <= {0, 1}


def test_build_presence_absence_matrix_uses_supplied_pi_vector():
    result = build_presence_absence_matrix(
        n_genomes=4,
        n_genes=3,
        pi_vector=[1.0, 0.0, 1.0],
        rng=1,
    )

    assert result["sim_gene_0"].eq(1).all()
    assert result["sim_gene_1"].eq(0).all()
    assert result["sim_gene_2"].eq(1).all()


def test_build_presence_absence_matrix_defaults_to_half_prevalence():
    result = build_presence_absence_matrix(n_genomes=4, n_genes=3, rng=0)

    expected = pd.DataFrame(
        {
            "sim_gene_0": [0, 1, 0, 0],
            "sim_gene_1": [1, 0, 0, 0],
            "sim_gene_2": [1, 0, 0, 1],
        },
        index=[f"sim_genome_{i}" for i in range(4)],
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


def test_build_observed_data_matrix_masks_present_genes_using_completeness_series():
    data = pd.DataFrame(
        {
            "gene_1": [1, 1],
            "gene_2": [1, 0],
            "gene_3": [0, 1],
        },
        index=["genome_1", "genome_2"],
    )
    completeness = pd.Series([1.0, 0.0], index=data.index)

    result = build_observed_data_matrix(data, completeness, rng=0)

    expected = pd.DataFrame(
        {
            "gene_1": [1, 0],
            "gene_2": [1, 0],
            "gene_3": [0, 0],
        },
        index=data.index,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_build_observed_data_matrix_accepts_array_like_completeness_levels():
    data = pd.DataFrame(
        {
            "gene_1": [1, 1, 0],
            "gene_2": [0, 1, 1],
        },
        index=["genome_1", "genome_2", "genome_3"],
    )

    result = build_observed_data_matrix(data, [1.0, 0.0, 1.0], rng=0)

    expected = pd.DataFrame(
        {
            "gene_1": [1, 0, 0],
            "gene_2": [0, 0, 1],
        },
        index=data.index,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_build_observed_data_matrix_returns_boolean_observation_mask():
    data = pd.DataFrame(
        {
            "gene_1": [1, 1],
            "gene_2": [0, 1],
            "gene_3": [1, 0],
        },
        index=["genome_1", "genome_2"],
    )

    observed, mask = build_observed_data_matrix(
        data,
        [1.0, 0.0],
        rng=0,
        return_observation_mask=True,
    )

    expected_observed = pd.DataFrame(
        {
            "gene_1": [1, 0],
            "gene_2": [0, 0],
            "gene_3": [1, 0],
        },
        index=data.index,
    )
    expected_mask = pd.DataFrame(
        {
            "gene_1": [True, False],
            "gene_2": [True, False],
            "gene_3": [True, False],
        },
        index=data.index,
    )

    pd.testing.assert_frame_equal(observed, expected_observed)
    pd.testing.assert_frame_equal(mask, expected_mask)
    assert mask.dtypes.eq(bool).all()


def test_build_observed_data_matrix_applies_gene_biases_without_enforcing_completeness():
    data = pd.DataFrame(
        {"gene_1": [1], "gene_2": [1], "gene_3": [1]},
        index=["genome_1"],
    )
    gene_biases = pd.Series([1.0, 0.2, 1.0], index=data.columns)

    observed = build_observed_data_matrix(
        data,
        [1.0],
        rng=0,
        gene_biases=gene_biases,
    )

    expected = pd.DataFrame({"gene_1": [1], "gene_2": [0], "gene_3": [1]}, index=data.index)
    pd.testing.assert_frame_equal(observed, expected)


def test_build_observed_data_matrix_enforces_expected_observed_count():
    data = pd.DataFrame(
        {"gene_1": [1], "gene_2": [1], "gene_3": [1], "gene_4": [0]},
        index=["genome_1"],
    )
    gene_biases = pd.Series([1.0, 0.5, 0.25, 1.0], index=data.columns)

    observed, mask = build_observed_data_matrix(
        data,
        [2 / 3],
        rng=0,
        gene_biases=gene_biases,
        enforce_completeness=True,
        return_observation_mask=True,
    )

    assert observed.loc["genome_1"].sum() == 2
    assert mask.loc["genome_1"].sum() == 2
    assert observed.loc["genome_1", "gene_4"] == 0
    assert not mask.loc["genome_1", "gene_4"]
    assert observed.le(data).all().all()


def test_build_observed_data_matrix_handles_enforced_completeness_with_no_true_presence():
    data = pd.DataFrame({"gene_1": [0], "gene_2": [0]}, index=["genome_1"])

    observed, mask = build_observed_data_matrix(
        data,
        [1.0],
        rng=0,
        enforce_completeness=True,
        return_observation_mask=True,
    )

    expected_observed = pd.DataFrame({"gene_1": [0], "gene_2": [0]}, index=data.index)
    expected_mask = pd.DataFrame({"gene_1": [False], "gene_2": [False]}, index=data.index)

    pd.testing.assert_frame_equal(observed, expected_observed)
    pd.testing.assert_frame_equal(mask, expected_mask)


def test_build_observed_data_matrix_rejects_invalid_inputs():
    data = pd.DataFrame({"gene_1": [1], "gene_2": [2]}, index=["genome_1"])

    with pytest.raises(ValueError, match="must be binary"):
        build_observed_data_matrix(data, [1.0], rng=0)

    valid_data = pd.DataFrame({"gene_1": [1, 0]}, index=["genome_1", "genome_2"])

    with pytest.raises(ValueError, match="must match the matrix rows"):
        build_observed_data_matrix(
            valid_data,
            pd.Series([1.0, 0.5], index=["genome_1", "genome_x"]),
            rng=0,
        )

    with pytest.raises(ValueError, match="one value per matrix row"):
        build_observed_data_matrix(valid_data, [1.0], rng=0)

    with pytest.raises(ValueError, match="fall between 0 and 1"):
        build_observed_data_matrix(valid_data, [1.0, 1.2], rng=0)


def test_build_observed_data_matrix_rejects_misaligned_gene_biases_series():
    data = pd.DataFrame({"gene_1": [1], "gene_2": [0]}, index=["genome_1"])
    gene_biases = pd.Series([1.0, 0.5], index=["gene_1", "gene_x"])

    with pytest.raises(ValueError, match="must match the matrix columns"):
        build_observed_data_matrix(data, [1.0], rng=0, gene_biases=gene_biases)


def test_build_observed_data_matrix_rejects_gene_biases_with_wrong_length():
    data = pd.DataFrame({"gene_1": [1], "gene_2": [0]}, index=["genome_1"])

    with pytest.raises(ValueError, match="one value per matrix column"):
        build_observed_data_matrix(data, [1.0], rng=0, gene_biases=[1.0])


@pytest.mark.parametrize("gene_biases", ([0.0, 1.0], [1.2, 1.0]))
def test_build_observed_data_matrix_rejects_out_of_range_gene_biases(gene_biases):
    data = pd.DataFrame({"gene_1": [1], "gene_2": [0]}, index=["genome_1"])

    with pytest.raises(ValueError, match=r"range \(0, 1\]"):
        build_observed_data_matrix(data, [1.0], rng=0, gene_biases=gene_biases)


def test_build_observed_data_matrix_matches_transposed_legacy_fixture():
    data = pd.DataFrame(
        {
            "gene_1": [1, 1],
            "gene_2": [1, 0],
            "gene_3": [0, 1],
        },
        index=["genome_1", "genome_2"],
    )

    result = build_observed_data_matrix(data, [1.0, 0.0], rng=0)

    expected = pd.DataFrame(
        {
            "gene_1": [1, 0],
            "gene_2": [1, 0],
            "gene_3": [0, 0],
        },
        index=data.index,
    )
    pd.testing.assert_frame_equal(result, expected)
