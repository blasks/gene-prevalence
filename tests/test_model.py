import pandas as pd
import pytest

from gene_prevalence.model import BernoulliPrevalenceModel, PrevalenceFitResult


def test_fit_unbiased_returns_expected_prevalence():
    data = pd.DataFrame(
        {
            "genome_1": [1, 0],
            "genome_2": [1, 1],
        },
        index=["gene_1", "gene_2"],
    )
    lambdas = pd.Series([1.0, 0.5], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(
        estimator="unbiased",
        interval_method=None,
        alpha=None,
        clip=True,
    )

    expected = pd.Series([1.0, 1.0], index=data.index)
    pd.testing.assert_series_equal(result.prevalence, expected)
    assert result.estimator == "unbiased"
    assert result.interval_method is None
    assert result.alpha is None
    assert result.metadata == {}


def test_fit_unbiased_clips_values_when_requested():
    data = pd.DataFrame(
        {
            "genome_1": [1],
            "genome_2": [1],
        },
        index=["gene_1"],
    )
    lambdas = pd.Series([0.5, 0.5], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(
        estimator="unbiased",
        interval_method=None,
        alpha=None,
        clip=True,
    )

    expected = pd.Series([1.0], index=data.index)
    pd.testing.assert_series_equal(result.prevalence, expected)


def test_fit_unbiased_allows_unclipped_values():
    data = pd.DataFrame(
        {
            "genome_1": [1],
            "genome_2": [1],
        },
        index=["gene_1"],
    )
    lambdas = pd.Series([0.5, 0.5], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(
        estimator="unbiased",
        interval_method=None,
        alpha=None,
        clip=False,
    )

    expected = pd.Series([2.0], index=data.index)
    pd.testing.assert_series_equal(result.prevalence, expected)


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


def test_fit_with_chebyshev_returns_expected_interval_and_variance():
    data = pd.DataFrame(
        {
            "genome_1": [1, 0],
            "genome_2": [0, 1],
        },
        index=["gene_1", "gene_2"],
    )
    lambdas = pd.Series([1.0, 0.5], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(
        estimator="unbiased",
        interval_method="chebyshev",
        alpha=0.25,
        clip=True,
    )

    expected_prevalence = pd.Series([0.5, 1.0], index=data.index)
    expected_variance = pd.Series([0.25, 0.25], index=data.index)
    expected_lower = pd.Series([0.0, 0.0], index=data.index)
    expected_upper = pd.Series([1.0, 1.0], index=data.index)

    pd.testing.assert_series_equal(result.prevalence, expected_prevalence)
    pd.testing.assert_series_equal(result.metadata["variance"], expected_variance)
    lower, upper = result.confidence_interval()
    pd.testing.assert_series_equal(lower, expected_lower)
    pd.testing.assert_series_equal(upper, expected_upper)


def test_fit_with_chebyshev_can_return_unclipped_bounds():
    data = pd.DataFrame(
        {
            "genome_1": [1],
            "genome_2": [1],
        },
        index=["gene_1"],
    )
    lambdas = pd.Series([0.5, 0.5], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    result = model.fit(
        estimator="unbiased",
        interval_method="chebyshev",
        alpha=0.25,
        clip=False,
    )

    expected_prevalence = pd.Series([2.0], index=data.index)
    expected_variance = pd.Series([0.0], index=data.index)
    expected_lower = pd.Series([2.0], index=data.index)
    expected_upper = pd.Series([2.0], index=data.index)

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
        {
            "prevalence": [0.2],
            "ci_lower": [0.1],
            "ci_upper": [0.3],
            "variance": [0.01],
        },
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
    data = pd.DataFrame({"genome_1": [1]}, index=["gene_1"])
    lambdas = pd.Series([1.0], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    with pytest.raises(ValueError, match="alpha must be None"):
        model.fit(
            estimator="unbiased",
            interval_method=None,
            alpha=0.05,
            clip=True,
        )

    with pytest.raises(ValueError, match="alpha must be provided"):
        model.fit(
            estimator="unbiased",
            interval_method="chebyshev",
            alpha=None,
            clip=True,
        )

    with pytest.raises(ValueError, match="alpha must be strictly between 0 and 1"):
        model.fit(
            estimator="unbiased",
            interval_method="chebyshev",
            alpha=0.0,
            clip=True,
        )


def test_fit_rejects_invalid_estimator_interval_pairing():
    data = pd.DataFrame({"genome_1": [1]}, index=["gene_1"])
    lambdas = pd.Series([1.0], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    with pytest.raises(ValueError, match="not supported"):
        model.fit(
            estimator="unbiased",
            interval_method="likelihood_ratio",
            alpha=0.05,
            clip=True,
        )


def test_fit_rejects_unknown_estimator_and_interval_method():
    data = pd.DataFrame({"genome_1": [1]}, index=["gene_1"])
    lambdas = pd.Series([1.0], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    with pytest.raises(ValueError, match="Unsupported estimator"):
        model.fit(
            estimator="median",
            interval_method=None,
            alpha=None,
            clip=True,
        )

    with pytest.raises(ValueError, match="Unsupported interval_method"):
        model.fit(
            estimator="unbiased",
            interval_method="wald",
            alpha=0.05,
            clip=True,
        )


def test_fit_rejects_non_boolean_clip():
    data = pd.DataFrame({"genome_1": [1]}, index=["gene_1"])
    lambdas = pd.Series([1.0], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    with pytest.raises(ValueError, match="clip must be a boolean"):
        model.fit(
            estimator="unbiased",
            interval_method=None,
            alpha=None,
            clip=1,
        )


def test_fit_mle_is_reserved_but_not_implemented():
    data = pd.DataFrame({"genome_1": [1]}, index=["gene_1"])
    lambdas = pd.Series([1.0], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    with pytest.raises(NotImplementedError, match="mle"):
        model.fit(
            estimator="mle",
            interval_method=None,
            alpha=None,
            clip=True,
        )


def test_private_prevalence_dispatch_rejects_unknown_estimator():
    data = pd.DataFrame({"genome_1": [1]}, index=["gene_1"])
    lambdas = pd.Series([1.0], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)

    with pytest.raises(ValueError, match="Unsupported estimator"):
        model._estimate_prevalence("unknown")


def test_private_interval_dispatch_rejects_unknown_interval_method():
    data = pd.DataFrame({"genome_1": [1]}, index=["gene_1"])
    lambdas = pd.Series([1.0], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)
    prevalence = pd.Series([1.0], index=data.index)

    with pytest.raises(ValueError, match="Unsupported interval_method"):
        model._estimate_interval(
            estimator="unbiased",
            interval_method="unknown",
            prevalence=prevalence,
            alpha=0.05,
        )


def test_private_interval_dispatch_reserves_likelihood_ratio():
    data = pd.DataFrame({"genome_1": [1]}, index=["gene_1"])
    lambdas = pd.Series([1.0], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)
    prevalence = pd.Series([1.0], index=data.index)

    with pytest.raises(NotImplementedError, match="likelihood_ratio"):
        model._estimate_interval(
            estimator="mle",
            interval_method="likelihood_ratio",
            prevalence=prevalence,
            alpha=0.05,
        )


def test_private_chebyshev_rejects_non_unbiased_estimator():
    data = pd.DataFrame({"genome_1": [1]}, index=["gene_1"])
    lambdas = pd.Series([1.0], index=data.columns)
    model = BernoulliPrevalenceModel(data=data, lambdas=lambdas)
    prevalence = pd.Series([1.0], index=data.index)

    with pytest.raises(ValueError, match="requires estimator='unbiased'"):
        model._ci_chebyshev(
            estimator="mle",
            prevalence=prevalence,
            alpha=0.05,
        )
