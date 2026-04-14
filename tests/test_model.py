import pandas as pd
import pytest

from gene_prevalence.model import BernoulliPrevalenceModel


def test_estimate_prevalence_returns_expected_values():
    data = pd.DataFrame(
        {
            "genome_1": [1, 0],
            "genome_2": [1, 1],
        },
        index=["gene_1", "gene_2"],
    )
    lambdas = pd.Series([1.0, 0.5], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.estimate_prevalence()

    expected = pd.Series([1.0, 1.0], index=data.index)
    pd.testing.assert_series_equal(result, expected)
    pd.testing.assert_series_equal(model.pi_estimates_, expected)


def test_estimate_prevalence_clips_values_when_requested():
    data = pd.DataFrame(
        {
            "genome_1": [1],
            "genome_2": [1],
        },
        index=["gene_1"],
    )
    lambdas = pd.Series([0.5, 0.5], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas, clip_max=1.0)

    result = model.estimate_prevalence()

    expected = pd.Series([1.0], index=data.index)
    pd.testing.assert_series_equal(result, expected)
    pd.testing.assert_series_equal(model.pi_estimates_, expected)


def test_estimate_prevalence_allows_unclipped_values():
    data = pd.DataFrame(
        {
            "genome_1": [1],
            "genome_2": [1],
        },
        index=["gene_1"],
    )
    lambdas = pd.Series([0.5, 0.5], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas, clip_max=None)

    result = model.estimate_prevalence()

    expected = pd.Series([2.0], index=data.index)
    pd.testing.assert_series_equal(result, expected)
    pd.testing.assert_series_equal(model.pi_estimates_, expected)


def test_model_requires_matching_lambda_index():
    data = pd.DataFrame(
        {
            "genome_1": [1, 0],
            "genome_2": [0, 1],
        },
        index=["gene_1", "gene_2"],
    )
    lambdas = pd.Series([1.0, 0.5], index=["genome_1", "genome_x"])

    with pytest.raises(ValueError, match="index of lambdas"):
        BernoulliPrevalenceModel(data=data, lambdas=lambdas)


def test_model_rejects_non_binary_data():
    data = pd.DataFrame(
        {
            "genome_1": [1, 2],
            "genome_2": [0, 1],
        },
        index=["gene_1", "gene_2"],
    )
    lambdas = pd.Series([1.0, 0.5], index=data.columns)

    with pytest.raises(ValueError, match="Data must be binary"):
        BernoulliPrevalenceModel(data=data, lambdas=lambdas)


def test_estimate_variance_uses_stored_pi_estimates():
    data = pd.DataFrame(
        {
            "genome_1": [1, 0],
            "genome_2": [0, 1],
            "genome_3": [0, 1],
        },
        index=["gene_1", "gene_2"],
    )
    lambdas = pd.Series([0.5, 1.0, 0.25], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)
    model.pi_estimates_ = pd.Series([0.2, 0.5], index=data.index)

    result = model.estimate_variance()

    n = len(lambdas)
    expected = pd.Series(
        [
            (0.2 / (n**2)) * ((1 / 0.5) + (1 / 1.0) + (1 / 0.25) - (n * 0.2)),
            (0.5 / (n**2)) * ((1 / 0.5) + (1 / 1.0) + (1 / 0.25) - (n * 0.5)),
        ],
        index=data.index,
    )
    pd.testing.assert_series_equal(result, expected)
    pd.testing.assert_series_equal(model.variance_, expected)


def test_estimate_variance_estimates_prevalence_when_missing():
    data = pd.DataFrame(
        {
            "genome_1": [1],
            "genome_2": [1],
        },
        index=["gene_1"],
    )
    lambdas = pd.Series([0.5, 0.5], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas, clip_max=1.0)

    result = model.estimate_variance()

    expected_prevalence = pd.Series([1.0], index=data.index)
    expected_variance = pd.Series(
        [(1.0 / (2**2)) * ((1 / 0.5) + (1 / 0.5) - (2 * 1.0))],
        index=data.index,
    )
    pd.testing.assert_series_equal(model.pi_estimates_, expected_prevalence)
    pd.testing.assert_series_equal(result, expected_variance)
    pd.testing.assert_series_equal(model.variance_, expected_variance)


def test_confidence_interval_uses_stored_variance():
    data = pd.DataFrame(
        {
            "genome_1": [1, 0],
            "genome_2": [0, 1],
        },
        index=["gene_1", "gene_2"],
    )
    lambdas = pd.Series([1.0, 0.5], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)
    model.variance_ = pd.Series([0.04, 0.09], index=data.index)

    result = model.confidence_interval(alpha=0.25)

    expected = pd.Series([0.4, 0.6], index=data.index)
    pd.testing.assert_series_equal(result, expected)
    pd.testing.assert_series_equal(model.confidence_intervals_, expected)


def test_confidence_interval_estimates_variance_when_missing():
    data = pd.DataFrame(
        {
            "genome_1": [1],
            "genome_2": [1],
        },
        index=["gene_1"],
    )
    lambdas = pd.Series([0.5, 0.5], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas, clip_max=1.0)

    result = model.confidence_interval(alpha=0.25)

    expected_variance = pd.Series([0.5], index=data.index)
    expected = pd.Series([2**0.5], index=data.index)
    pd.testing.assert_series_equal(model.variance_, expected_variance)
    pd.testing.assert_series_equal(result, expected)
    pd.testing.assert_series_equal(model.confidence_intervals_, expected)


def test_confidence_interval_requires_alpha_between_zero_and_one():
    data = pd.DataFrame(
        {
            "genome_1": [1],
        },
        index=["gene_1"],
    )
    lambdas = pd.Series([1.0], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    with pytest.raises(ValueError, match="alpha must be strictly between 0 and 1"):
        model.confidence_interval(alpha=0.0)

    with pytest.raises(ValueError, match="alpha must be strictly between 0 and 1"):
        model.confidence_interval(alpha=1.0)
