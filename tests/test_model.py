import math

import pandas as pd
import pytest
from scipy.stats import chi2

from gene_prevalence.model import BernoulliPrevalenceModel, PrevalenceFitResult


def test_fit_unbiased_returns_expected_prevalence():
    data = pd.DataFrame(
        {"gene_1": [1, 1], "gene_2": [0, 1]},
        index=["genome_1", "genome_2"],
    )
    lambdas = pd.Series([1.0, 0.5], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(estimator="unbiased", interval_method=None, alpha=None, clip=True)

    expected = pd.Series([1.0, 1.0], index=data.columns)
    pd.testing.assert_series_equal(result.prevalence, expected)
    assert result.estimator == "unbiased"
    assert result.interval_method is None
    assert result.alpha is None
    assert result.metadata == {}


def test_fit_unbiased_clips_values_when_requested():
    data = pd.DataFrame({"gene_1": [1, 1]}, index=["genome_1", "genome_2"])
    lambdas = pd.Series([0.5, 0.5], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(estimator="unbiased", interval_method=None, alpha=None, clip=True)

    expected = pd.Series([1.0], index=data.columns)
    pd.testing.assert_series_equal(result.prevalence, expected)


def test_fit_unbiased_allows_unclipped_values():
    data = pd.DataFrame({"gene_1": [1, 1]}, index=["genome_1", "genome_2"])
    lambdas = pd.Series([0.5, 0.5], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(estimator="unbiased", interval_method=None, alpha=None, clip=False)

    expected = pd.Series([2.0], index=data.columns)
    pd.testing.assert_series_equal(result.prevalence, expected)


def test_model_requires_matching_lambda_index():
    data = pd.DataFrame(
        {"gene_1": [1, 0], "gene_2": [0, 1]},
        index=["genome_1", "genome_2"],
    )
    lambdas = pd.Series([1.0, 0.5], index=["genome_1", "gene_1"])

    with pytest.raises(ValueError, match="index of lambdas"):
        BernoulliPrevalenceModel(data=data, lambdas=lambdas)


def test_model_rejects_non_binary_data():
    data = pd.DataFrame(
        {"gene_1": [1, 2], "gene_2": [0, 1]},
        index=["genome_1", "genome_2"],
    )
    lambdas = pd.Series([1.0, 0.5], index=data.index)

    with pytest.raises(ValueError, match="Data must be binary"):
        BernoulliPrevalenceModel(data=data, lambdas=lambdas)


def test_fit_with_chebyshev_returns_expected_interval_and_variance():
    data = pd.DataFrame(
        {"gene_1": [1, 0], "gene_2": [0, 1]},
        index=["genome_1", "genome_2"],
    )
    lambdas = pd.Series([1.0, 0.5], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(
        estimator="unbiased",
        interval_method="chebyshev",
        alpha=0.25,
        clip=True,
    )

    expected_prevalence = pd.Series([0.5, 1.0], index=data.columns)
    expected_variance = pd.Series([0.25, 0.25], index=data.columns)
    expected_lower = pd.Series([0.0, 0.0], index=data.columns)
    expected_upper = pd.Series([1.0, 1.0], index=data.columns)

    pd.testing.assert_series_equal(result.prevalence, expected_prevalence)
    pd.testing.assert_series_equal(result.metadata["variance"], expected_variance)
    lower, upper = result.confidence_interval()
    pd.testing.assert_series_equal(lower, expected_lower)
    pd.testing.assert_series_equal(upper, expected_upper)


def test_fit_with_chebyshev_can_return_unclipped_bounds():
    data = pd.DataFrame({"gene_1": [1, 1]}, index=["genome_1", "genome_2"])
    lambdas = pd.Series([0.5, 0.5], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(
        estimator="unbiased",
        interval_method="chebyshev",
        alpha=0.25,
        clip=False,
    )

    expected_prevalence = pd.Series([2.0], index=data.columns)
    expected_variance = pd.Series([0.0], index=data.columns)
    expected_lower = pd.Series([2.0], index=data.columns)
    expected_upper = pd.Series([2.0], index=data.columns)

    pd.testing.assert_series_equal(result.prevalence, expected_prevalence)
    pd.testing.assert_series_equal(result.metadata["variance"], expected_variance)
    lower, upper = result.confidence_interval()
    pd.testing.assert_series_equal(lower, expected_lower)
    pd.testing.assert_series_equal(upper, expected_upper)


def test_summary_includes_prevalence_interval_and_variance():
    result = PrevalenceFitResult(
        prevalence=pd.Series([0.2], index=["gene_1"]),
        estimator="unbiased",
        interval_method="chebyshev",
        alpha=0.05,
        ci_lower=pd.Series([0.1], index=["gene_1"]),
        ci_upper=pd.Series([0.3], index=["gene_1"]),
        metadata={"variance": pd.Series([0.01], index=["gene_1"])},
    )

    summary = result.summary()

    expected = pd.DataFrame(
        {"prevalence": [0.2], "ci_lower": [0.1], "ci_upper": [0.3], "variance": [0.01]},
        index=["gene_1"],
    )
    pd.testing.assert_frame_equal(summary, expected)


def test_summary_with_prevalence_only_returns_single_column():
    result = PrevalenceFitResult(
        prevalence=pd.Series([0.2], index=["gene_1"]),
        estimator="unbiased",
        interval_method=None,
        alpha=None,
    )

    summary = result.summary()

    expected = pd.DataFrame({"prevalence": [0.2]}, index=["gene_1"])
    pd.testing.assert_frame_equal(summary, expected)


def test_confidence_interval_raises_when_no_interval_was_fit():
    result = PrevalenceFitResult(
        prevalence=pd.Series([0.2], index=["gene_1"]),
        estimator="unbiased",
        interval_method=None,
        alpha=None,
    )

    with pytest.raises(ValueError, match="No confidence interval"):
        result.confidence_interval()


def test_fit_rejects_invalid_alpha_combinations():
    data = pd.DataFrame({"gene_1": [1]}, index=["genome_1"])
    lambdas = pd.Series([1.0], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    with pytest.raises(ValueError, match="alpha must be None"):
        model.fit(estimator="unbiased", interval_method=None, alpha=0.05, clip=True)

    with pytest.raises(ValueError, match="alpha must be provided"):
        model.fit(estimator="unbiased", interval_method="chebyshev", alpha=None, clip=True)

    with pytest.raises(ValueError, match="alpha must be strictly between 0 and 1"):
        model.fit(estimator="unbiased", interval_method="chebyshev", alpha=0.0, clip=True)


def test_fit_rejects_invalid_estimator_interval_pairing():
    data = pd.DataFrame({"gene_1": [1]}, index=["genome_1"])
    lambdas = pd.Series([1.0], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    with pytest.raises(ValueError, match="not supported"):
        model.fit(
            estimator="unbiased",
            interval_method="likelihood_ratio",
            alpha=0.05,
            clip=True,
        )


def test_fit_rejects_unknown_estimator_and_interval_method():
    data = pd.DataFrame({"gene_1": [1]}, index=["genome_1"])
    lambdas = pd.Series([1.0], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    with pytest.raises(ValueError, match="Unsupported estimator"):
        model.fit(estimator="median", interval_method=None, alpha=None, clip=True)

    with pytest.raises(ValueError, match="Unsupported interval_method"):
        model.fit(estimator="unbiased", interval_method="wald", alpha=0.05, clip=True)


def test_fit_rejects_non_boolean_clip():
    data = pd.DataFrame({"gene_1": [1]}, index=["genome_1"])
    lambdas = pd.Series([1.0], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    with pytest.raises(ValueError, match="clip must be a boolean"):
        model.fit(estimator="unbiased", interval_method=None, alpha=None, clip=1)


def test_fit_mle_returns_expected_result_shape():
    data = pd.DataFrame(
        {"gene_1": [1, 1], "gene_2": [0, 1]},
        index=["genome_1", "genome_2"],
    )
    lambdas = pd.Series([1.0, 1.0], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(estimator="mle", interval_method=None, alpha=None, clip=True)

    expected = pd.Series([1.0, 0.5], index=data.columns)
    pd.testing.assert_series_equal(result.prevalence, expected)
    assert result.estimator == "mle"
    assert result.interval_method is None
    assert result.alpha is None
    assert result.metadata == {}


def test_fit_mle_matches_sample_mean_when_all_lambdas_are_one():
    data = pd.DataFrame(
        {
            "gene_1": [1, 0, 1, 0],
            "gene_2": [0, 1, 1, 1],
        },
        index=["genome_1", "genome_2", "genome_3", "genome_4"],
    )
    lambdas = pd.Series([1.0, 1.0, 1.0, 1.0], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(estimator="mle", interval_method=None, alpha=None, clip=False)

    expected = data.mean(axis=0).astype(float)
    pd.testing.assert_series_equal(result.prevalence, expected)


def test_fit_mle_differs_from_unbiased_estimator_for_incomplete_genomes():
    data = pd.DataFrame({"gene_1": [1, 0, 0]}, index=["genome_1", "genome_2", "genome_3"])
    lambdas = pd.Series([1.0, 0.5, 0.5], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    mle_result = model.fit(estimator="mle", interval_method=None, alpha=None, clip=False)
    unbiased_result = model.fit(estimator="unbiased", interval_method=None, alpha=None, clip=False)

    mle_value = mle_result.prevalence.iloc[0]
    unbiased_value = unbiased_result.prevalence.iloc[0]
    assert 0.0 <= mle_value <= 1.0
    assert mle_value != pytest.approx(unbiased_value)


def test_fit_mle_hits_zero_boundary_for_all_zero_observations():
    data = pd.DataFrame({"gene_1": [0, 0, 0]}, index=["genome_1", "genome_2", "genome_3"])
    lambdas = pd.Series([1.0, 0.7, 0.4], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(estimator="mle", interval_method=None, alpha=None, clip=False)

    expected = pd.Series([0.0], index=data.columns)
    pd.testing.assert_series_equal(result.prevalence, expected)


def test_fit_mle_hits_one_boundary_for_all_one_observations_with_full_completeness():
    data = pd.DataFrame({"gene_1": [1, 1, 1]}, index=["genome_1", "genome_2", "genome_3"])
    lambdas = pd.Series([1.0, 1.0, 1.0], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(estimator="mle", interval_method=None, alpha=None, clip=False)

    expected = pd.Series([1.0], index=data.columns)
    pd.testing.assert_series_equal(result.prevalence, expected)


def test_fit_mle_remains_numerically_stable_with_mixed_incomplete_genomes():
    data = pd.DataFrame(
        {
            "gene_1": [1, 0, 1, 0],
            "gene_2": [0, 1, 0, 0],
        },
        index=["genome_1", "genome_2", "genome_3", "genome_4"],
    )
    lambdas = pd.Series([1.0, 0.8, 0.3, 0.2], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(estimator="mle", interval_method=None, alpha=None, clip=False)

    assert result.prevalence.between(0.0, 1.0).all()
    assert result.prevalence.notna().all()


def test_fit_mle_clip_is_effectively_a_no_op():
    data = pd.DataFrame(
        {
            "gene_1": [1, 1, 0],
            "gene_2": [0, 1, 0],
        },
        index=["genome_1", "genome_2", "genome_3"],
    )
    lambdas = pd.Series([1.0, 0.6, 0.4], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    unclipped = model.fit(estimator="mle", interval_method=None, alpha=None, clip=False)
    clipped = model.fit(estimator="mle", interval_method=None, alpha=None, clip=True)

    pd.testing.assert_series_equal(clipped.prevalence, unclipped.prevalence)


def test_private_prevalence_dispatch_rejects_unknown_estimator():
    data = pd.DataFrame({"gene_1": [1]}, index=["genome_1"])
    lambdas = pd.Series([1.0], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    with pytest.raises(ValueError, match="Unsupported estimator"):
        model._estimate_prevalence("unknown")


def test_private_interval_dispatch_rejects_unknown_interval_method():
    data = pd.DataFrame({"gene_1": [1]}, index=["genome_1"])
    lambdas = pd.Series([1.0], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)
    prevalence = pd.Series([1.0], index=data.columns)

    with pytest.raises(ValueError, match="Unsupported interval_method"):
        model._estimate_interval(
            estimator="unbiased",
            interval_method="unknown",
            prevalence=prevalence,
            alpha=0.05,
        )


def test_fit_with_likelihood_ratio_returns_bounded_intervals_and_metadata():
    data = pd.DataFrame(
        {
            "gene_1": [1, 0, 1, 0],
            "gene_2": [0, 1, 0, 0],
            "gene_3": [1, 1, 0, 1],
        },
        index=["genome_1", "genome_2", "genome_3", "genome_4"],
    )
    lambdas = pd.Series([1.0, 0.8, 0.6, 0.4], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(
        estimator="mle",
        interval_method="likelihood_ratio",
        alpha=0.05,
        clip=True,
    )

    lower, upper = result.confidence_interval()
    assert lower.between(0.0, 1.0).all()
    assert upper.between(0.0, 1.0).all()
    assert ((lower <= result.prevalence) & (result.prevalence <= upper)).all()
    assert result.metadata["cutoff"] == pytest.approx(float(chi2.ppf(0.95, df=1)))
    assert result.metadata["solver"] == "root_scalar_brentq"


def test_fit_with_likelihood_ratio_handles_boundary_mles():
    data = pd.DataFrame(
        {
            "all_zero": [0, 0, 0],
            "all_one": [1, 1, 1],
            "interior": [1, 0, 1],
        },
        index=["genome_1", "genome_2", "genome_3"],
    )
    lambdas = pd.Series([1.0, 1.0, 1.0], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(
        estimator="mle",
        interval_method="likelihood_ratio",
        alpha=0.05,
        clip=False,
    )

    lower, upper = result.confidence_interval()
    assert result.prevalence.loc["all_zero"] == pytest.approx(0.0)
    assert lower.loc["all_zero"] == pytest.approx(0.0)
    assert 0.0 < upper.loc["all_zero"] < 1.0
    assert result.prevalence.loc["all_one"] == pytest.approx(1.0)
    assert 0.0 < lower.loc["all_one"] < 1.0
    assert upper.loc["all_one"] == pytest.approx(1.0)


def test_fit_with_likelihood_ratio_returns_finite_ordered_interval_for_interior_mle():
    data = pd.DataFrame(
        {"gene_1": [1, 0, 1, 0]},
        index=["genome_1", "genome_2", "genome_3", "genome_4"],
    )
    lambdas = pd.Series([1.0, 0.8, 0.6, 0.4], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(
        estimator="mle",
        interval_method="likelihood_ratio",
        alpha=0.05,
        clip=False,
    )

    lower, upper = result.confidence_interval()
    assert 0.0 < result.prevalence.iloc[0] < 1.0
    assert 0.0 <= lower.iloc[0] <= result.prevalence.iloc[0] <= upper.iloc[0] <= 1.0
    assert lower.notna().all()
    assert upper.notna().all()


def test_negative_log_likelihood_matches_expected_formula_with_clipping():
    data = pd.DataFrame({"gene_1": [1, 0]}, index=["genome_1", "genome_2"])
    lambdas = pd.Series([1.0, 0.5], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    observations = data["gene_1"].to_numpy(dtype=float)
    value = model.negative_log_likelihood(0.0, observations)
    epsilon = model._MLE_EPSILON
    expected = -(math.log(epsilon) + math.log(1.0 - epsilon))

    assert value == pytest.approx(expected)
    assert value > 0.0


def test_private_interval_dispatch_runs_likelihood_ratio():
    data = pd.DataFrame(
        {"gene_1": [1, 0, 1]},
        index=["genome_1", "genome_2", "genome_3"],
    )
    lambdas = pd.Series([1.0, 0.8, 0.6], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)
    prevalence = model.fit(estimator="mle", interval_method=None, alpha=None, clip=False).prevalence

    lower, upper, metadata = model._estimate_interval(
        estimator="mle",
        interval_method="likelihood_ratio",
        prevalence=prevalence,
        alpha=0.05,
    )

    assert lower.index.equals(prevalence.index)
    assert upper.index.equals(prevalence.index)
    assert metadata["solver"] == "root_scalar_brentq"


def test_private_chebyshev_rejects_non_unbiased_estimator():
    data = pd.DataFrame({"gene_1": [1]}, index=["genome_1"])
    lambdas = pd.Series([1.0], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)
    prevalence = pd.Series([1.0], index=data.columns)

    with pytest.raises(ValueError, match="requires estimator='unbiased'"):
        model._ci_chebyshev(estimator="mle", prevalence=prevalence, alpha=0.05)


def test_ci_likelihood_ratio_return_prevalence_matches_plain_mle_fit():
    data = pd.DataFrame(
        {"gene_1": [1, 0, 1], "gene_2": [0, 1, 1]},
        index=["genome_1", "genome_2", "genome_3"],
    )
    lambdas = pd.Series([1.0, 0.8, 0.6], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    mle_fit = model.fit(estimator="mle", interval_method=None, alpha=None, clip=False)
    prevalence, lower, upper, metadata = model._ci_likelihood_ratio(
        alpha=0.05,
        return_prevalence=True,
    )

    pd.testing.assert_series_equal(prevalence, mle_fit.prevalence)
    assert ((lower <= prevalence) & (prevalence <= upper)).all()
    assert metadata["solver"] == "root_scalar_brentq"


def test_prevalence_outputs_remain_gene_indexed():
    data = pd.DataFrame(
        {"gene_1": [1, 0], "gene_2": [0, 1]},
        index=["genome_1", "genome_2"],
    )
    lambdas = pd.Series([1.0, 0.5], index=data.index)
    result = BernoulliPrevalenceModel(data=data, lambdas=lambdas).fit(
        estimator="unbiased",
        interval_method=None,
        alpha=None,
        clip=True,
    )

    assert result.prevalence.index.equals(data.columns)


def test_fit_results_match_transposed_legacy_fixture():
    data = pd.DataFrame(
        {"gene_1": [1, 1], "gene_2": [0, 1]},
        index=["genome_1", "genome_2"],
    )
    lambdas = pd.Series([1.0, 0.5], index=data.index)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    unbiased = model.fit(estimator="unbiased", interval_method=None, alpha=None, clip=True)
    mle = model.fit(estimator="mle", interval_method=None, alpha=None, clip=True)

    pd.testing.assert_series_equal(unbiased.prevalence, pd.Series([1.0, 1.0], index=data.columns))
    pd.testing.assert_series_equal(mle.prevalence, pd.Series([1.0, 0.5], index=data.columns))
