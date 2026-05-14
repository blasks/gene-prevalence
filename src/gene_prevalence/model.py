"""Bernoulli prevalence estimation model and fit results."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, root_scalar
from scipy.stats import chi2


@dataclass(frozen=True)
class ConfidenceIntervalResult:
    """
    Store one fitted confidence interval and any method-specific metadata.

    Parameters
    ----------
    method : {"chebyshev", "likelihood_ratio"}
        Name of the confidence interval method used for this interval.
    alpha : float
        Significance level used for confidence interval construction.
    ci_lower : pd.Series
        Lower confidence bounds indexed by gene family.
    ci_upper : pd.Series
        Upper confidence bounds indexed by gene family.
    metadata : dict[str, object]
        Additional interval-specific outputs.
    """

    method: str
    alpha: float
    ci_lower: pd.Series
    ci_upper: pd.Series
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PrevalenceFitResult:
    """
    Store the output of one prevalence estimation procedure.

    Parameters
    ----------
    prevalence : pd.Series
        Estimated prevalence values indexed by gene family.
    estimator : {"unbiased", "mle"}
        Name of the point estimator used to compute ``prevalence``.
    interval_method : {"chebyshev", "likelihood_ratio"} or None
        Name of the confidence interval method used for this fit, if any.
    alpha : float or None
        Significance level used for confidence interval construction. This is
        ``None`` when no interval was requested.
    ci_lower : pd.Series or None
        Lower confidence bounds indexed by gene family. This is ``None`` when
        no interval was requested.
    ci_upper : pd.Series or None
        Upper confidence bounds indexed by gene family. This is ``None`` when
        no interval was requested.
    metadata : dict[str, object]
        Additional fit-level outputs for the single-interval ``fit()`` path.
        For example, the unbiased estimator with the Chebyshev interval stores
        the variance estimates under ``metadata["variance"]``.
    confidence_intervals : list[ConfidenceIntervalResult]
        All confidence intervals computed for this fit, in request order.
        When exactly one interval was computed, the legacy ``interval_method``,
        ``alpha``, ``ci_lower``, and ``ci_upper`` fields mirror that interval
        for backward-compatible access.
    """

    prevalence: pd.Series
    estimator: str
    interval_method: str | None
    alpha: float | None
    ci_lower: pd.Series | None = None
    ci_upper: pd.Series | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    confidence_intervals: list[ConfidenceIntervalResult] = field(default_factory=list)

    def confidence_interval(
        self,
        method: str | None = None,
        alpha: float | None = None,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Return one fitted confidence interval.

        Parameters
        ----------
        method : {"chebyshev", "likelihood_ratio"} or None
            Confidence interval method to select. This must be provided
            together with ``alpha`` when multiple confidence intervals were
            computed. Leave as ``None`` to use the legacy single-interval
            behavior.
        alpha : float or None
            Significance level identifying the interval to select. This must
            be provided together with ``method``.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            Lower and upper confidence bounds indexed by gene family.

        Raises
        ------
        ValueError
            If this fit was created without a confidence interval, if multiple
            confidence intervals are available but no selector is provided, or
            if the requested interval does not exist.
        """
        if (method is None) != (alpha is None):
            raise ValueError("Both method and alpha must be provided together.")

        if method is None and alpha is None:
            if len(self.confidence_intervals) == 1:
                interval = self.confidence_intervals[0]
                return interval.ci_lower, interval.ci_upper

            if self.ci_lower is not None and self.ci_upper is not None:
                return self.ci_lower, self.ci_upper

            if len(self.confidence_intervals) > 1:
                raise ValueError(
                    "Multiple confidence intervals were computed; provide both method and "
                    "alpha to select one."
                )

            raise ValueError("No confidence interval was computed for this fit.")

        for interval in self.confidence_intervals:
            if interval.method == method and interval.alpha == alpha:
                return interval.ci_lower, interval.ci_upper

        raise ValueError(
            f"No confidence interval was computed for method={method!r} and alpha={alpha!r}."
        )

    def get_confidence_intervals(self) -> list[ConfidenceIntervalResult]:
        """
        Return all computed confidence intervals in request order.

        Returns
        -------
        list[ConfidenceIntervalResult]
            Shallow copy of the stored interval results.
        """
        return list(self.confidence_intervals)

    def summary(self) -> pd.DataFrame:
        """
        Return a tabular summary of the fitted quantities.

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by gene family. The ``prevalence`` column is
            always present. When exactly one confidence interval is exposed
            through the legacy single-interval fields, the ``ci_lower`` and
            ``ci_upper`` columns are included. If the fit metadata contains a
            series under the ``variance`` key, it is also included as a
            ``variance`` column.
        """
        summary = pd.DataFrame({"prevalence": self.prevalence})

        if self.ci_lower is not None:
            summary["ci_lower"] = self.ci_lower

        if self.ci_upper is not None:
            summary["ci_upper"] = self.ci_upper

        variance = self.metadata.get("variance")
        if isinstance(variance, pd.Series):
            summary["variance"] = variance

        return summary


class BernoulliPrevalenceModel:
    """
    Estimate gene family prevalence with a Bernoulli observation model.

    This model assumes an observed binary presence/absence matrix ``data``,
    where rows correspond to gene families and columns correspond to genomes.
    Each genome has an associated completeness parameter ``lambda_i`` in
    ``lambdas``.

    The Bernoulli model assumes:

    .. math::

        X_{ij} \\sim \\mathrm{Bernoulli}(\\pi_j \\lambda_i)

    where:

    - :math:`X_{ij}` is the observed presence/absence value for gene family
      ``j`` in genome ``i``
    - :math:`\\pi_j` is the true prevalence of gene family ``j``
    - :math:`\\lambda_i` is the completeness of genome ``i``

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing presence/absence data for gene families across
        genomes.
    lambdas : pd.Series
        A series of lambda values corresponding to the completeness of each
        genome in the dataset, with values between 0 and 1.

    Attributes
    ----------
    data : pd.DataFrame
        Binary presence/absence data indexed by gene family and genome.
    lambdas : pd.Series
        Genome completeness values aligned to ``data.columns``.
    """

    _SUPPORTED_ESTIMATORS = {"unbiased", "mle"}
    _SUPPORTED_INTERVAL_METHODS = {"chebyshev", "likelihood_ratio"}
    _ALLOWED_INTERVALS = {
        "unbiased": {"chebyshev"},
        "mle": {"likelihood_ratio"},
    }
    _MLE_EPSILON = 1e-8
    _MLE_MAXITER = 500
    _LR_SOLVER_METHOD = "brentq"

    def __init__(self, data, lambdas):
        self.data = data.astype(int)
        self.lambdas = lambdas
        self._validate_inputs()
        self._data_np = self.data.to_numpy(dtype=float)
        self._lambdas_np = self.lambdas.to_numpy(dtype=float)

    def fit(
        self,
        *,
        estimator: str,
        interval_method: str | None,
        alpha: float | None,
        clip: bool,
    ) -> PrevalenceFitResult:
        """
        Fit one prevalence estimator and, optionally, one confidence interval.

        The point estimator and interval method are specified independently so
        the API can support estimator-specific interval methods today and
        shared interval methods in the future. At present:

        - ``estimator="unbiased"`` may be paired with
          ``interval_method="chebyshev"``
        - ``estimator="mle"`` may be paired with
          ``interval_method="likelihood_ratio"``

        The unbiased point estimator is:

        .. math::

            \\hat{\\pi}_j = \\frac{1}{n} \\sum_{i=1}^{n} \\frac{X_{ij}}{\\lambda_i}

        where:

        - :math:`X_{ij}` is the observed binary presence/absence value for
          gene family ``j`` in genome ``i``
        - :math:`\\lambda_i` is the completeness of genome ``i``
        - :math:`n` is the number of genomes

        The maximum likelihood estimator (MLE) for gene family :math:`j`
        maximizes the Bernoulli likelihood

        .. math::

            L(\\pi_j \\mid X, \\lambda)
            = \\prod_{i=1}^{n}
            (\\pi_j \\lambda_i)^{X_{ij}}
            (1 - \\pi_j \\lambda_i)^{1 - X_{ij}}

        or, equivalently, the log-likelihood

        .. math::

            \\ell(\\pi_j \\mid X, \\lambda)
            = \\sum_{i=1}^{n}
            X_{ij} \\log(\\pi_j \\lambda_i)
            + (1 - X_{ij}) \\log(1 - \\pi_j \\lambda_i)

        The optimizer searches over :math:`0 \\leq \\pi_j \\leq 1`. To avoid
        undefined logarithms at the boundaries, the probabilities
        :math:`\\pi_j \\lambda_i` are clipped to
        :math:`[\\varepsilon, 1 - \\varepsilon]` before evaluating the
        objective.

        The Chebyshev confidence interval for the unbiased estimator is based
        on the variance:

        .. math::

            \\mathrm{Var}(\\hat{\\pi}_j)
            = n^{-2} \\sum_{i=1}^{n}
            \\left(1 - \\hat{\\pi}_j \\lambda_i\\right) \\frac{\\hat{\\pi}_j}{\\lambda_i}

        and interval width:

        .. math::

            \\mathrm{CI}_j = \\sqrt{\\frac{\\mathrm{Var}(\\hat{\\pi}_j)}{\\alpha}}

        The likelihood-ratio confidence interval for the MLE is the set of
        values satisfying:

        .. math::

            2 \\left\\{
            -\\ell(\\pi_j \\mid X, \\lambda)
            + \\ell(\\hat{\\pi}^{\\mathrm{MLE}}_j \\mid X, \\lambda)
            \\right\\}
            \\leq \\chi^2_{1, 1 - \\alpha}

        where :math:`\\chi^2_{1, 1 - \\alpha}` is the
        :math:`1 - \\alpha` quantile of a chi-square distribution with one
        degree of freedom.

        Parameters
        ----------
        estimator : {"unbiased", "mle"}
            Point estimator to use.
        interval_method : {"chebyshev", "likelihood_ratio"} or None
            Confidence interval method to use. Pass ``None`` to fit only the
            point estimator.
        alpha : float or None
            Significance level used for confidence interval construction.
            Provide a value satisfying ``0 < alpha < 1`` when
            ``interval_method`` is not ``None``. Pass ``None`` when no
            confidence interval is requested.
        clip : bool
            If ``True``, clip prevalence estimates and interval upper bounds
            to at most ``1``. If ``False``, return the raw estimator output.

        Returns
        -------
        PrevalenceFitResult
            The prevalence estimates and any fitted interval bounds. When an
            interval is requested, the returned result also stores one
            ``ConfidenceIntervalResult`` in ``confidence_intervals``.

        Raises
        ------
        ValueError
            If the estimator, interval method, or ``alpha`` value is invalid
            or if the requested estimator/interval pairing is not supported.
        NotImplementedError
            If a valid but currently unimplemented estimator or interval
            method is requested.
        """
        self._validate_fit_arguments(
            estimator=estimator,
            interval_method=interval_method,
            alpha=alpha,
            clip=clip,
        )
        intervals = []
        if interval_method is not None:
            intervals.append({"method": interval_method, "alpha": alpha})
        return self._fit_with_interval_specs(
            estimator=estimator,
            intervals=intervals,
            clip=clip,
            preserve_single_interval_metadata=True,
        )

    def fit_multiple_ci(
        self,
        *,
        estimator: str,
        intervals: list[dict[str, object]],
        clip: bool,
    ) -> PrevalenceFitResult:
        """
        Fit one prevalence estimator and compute multiple confidence intervals.

        Parameters
        ----------
        estimator : {"unbiased", "mle"}
            Point estimator to use.
        intervals : list[dict[str, object]]
            List of interval specifications. Each specification must contain a
            ``method`` and an ``alpha`` entry. Repeated methods with
            different alpha values are allowed.
        clip : bool
            If ``True``, clip prevalence estimates and interval upper bounds
            to at most ``1``. If ``False``, return the raw estimator output.

        Returns
        -------
        PrevalenceFitResult
            The prevalence estimates and all requested confidence intervals.
            If exactly one interval is requested, the legacy single-interval
            result fields are also populated for compatibility.
        """
        self._validate_estimator_and_clip(estimator=estimator, clip=clip)
        if not isinstance(intervals, list):
            raise ValueError("intervals must be provided as a list of interval specs.")

        validated_intervals = [
            self._validate_interval_spec(estimator=estimator, interval_spec=interval_spec)
            for interval_spec in intervals
        ]
        return self._fit_with_interval_specs(
            estimator=estimator,
            intervals=validated_intervals,
            clip=clip,
            preserve_single_interval_metadata=False,
        )

    def _validate_inputs(self):
        """
        Validate model inputs.

        Raises
        ------
        ValueError
            If the index of ``lambdas`` does not match ``data.columns`` or if
            ``data`` contains values other than 0 and 1.
        """
        if not self.lambdas.index.equals(self.data.columns):
            raise ValueError("The index of lambdas must match the columns of data.")

        if not self.data.map(lambda x: x in [0, 1]).all().all():
            raise ValueError("Data must be binary (0 and 1).")

    def _validate_fit_arguments(self, estimator, interval_method, alpha, clip):
        """Validate arguments passed to :meth:`fit`."""
        self._validate_estimator_and_clip(estimator=estimator, clip=clip)

        if interval_method is None:
            if alpha is not None:
                raise ValueError("alpha must be None when interval_method is None.")
        else:
            self._validate_interval_spec(
                estimator=estimator,
                interval_spec={"method": interval_method, "alpha": alpha},
                require_alpha=True,
            )

    def _validate_estimator_and_clip(self, estimator, clip):
        """Validate estimator and clip arguments shared across fit entrypoints."""
        if estimator not in self._SUPPORTED_ESTIMATORS:
            raise ValueError(
                f"Unsupported estimator {estimator!r}. "
                f"Choose from {sorted(self._SUPPORTED_ESTIMATORS)}."
            )

        if not isinstance(clip, bool):
            raise ValueError("clip must be a boolean value.")

    def _validate_interval_spec(
        self,
        *,
        estimator: str,
        interval_spec: dict[str, object],
        require_alpha: bool = True,
    ) -> dict[str, Any]:
        """Validate one interval specification and return its normalized values."""
        if not isinstance(interval_spec, dict) or not interval_spec:
            raise ValueError("Each interval spec must be a non-empty dictionary.")

        if "method" not in interval_spec:
            raise ValueError("Each interval spec must include a 'method'.")

        if "alpha" not in interval_spec:
            raise ValueError("Each interval spec must include an 'alpha'.")

        method = interval_spec["method"]
        alpha = interval_spec["alpha"]

        if method not in self._SUPPORTED_INTERVAL_METHODS:
            raise ValueError(
                f"Unsupported interval_method {method!r}. "
                f"Choose from {sorted(self._SUPPORTED_INTERVAL_METHODS)}."
            )

        if alpha is None:
            if require_alpha:
                raise ValueError("alpha must be provided when an interval method is requested.")
        elif not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1.")

        allowed_intervals = self._ALLOWED_INTERVALS[estimator]
        if method not in allowed_intervals:
            raise ValueError(
                f"interval_method={method!r} is not supported for estimator={estimator!r}."
            )

        return {"method": method, "alpha": alpha}

    def _estimate_prevalence(self, estimator):
        """Dispatch to the requested prevalence estimator and cacheable artifacts."""
        if estimator == "unbiased":
            return self._fit_unbiased()

        if estimator == "mle":
            return self._fit_mle()

        raise ValueError(f"Unsupported estimator {estimator!r}.")

    def _fit_unbiased(self):
        """
        Estimate gene family prevalence with the unbiased estimator.

        The estimator is:

        .. math::

            \\hat{\\pi}_j = \\frac{1}{n} \\sum_{i=1}^{n} \\frac{1}{\\lambda_i} X_{ij}

        where ``X_{ij}`` is the observed binary presence/absence value for
        gene family ``j`` in genome ``i``, and ``\\lambda_i`` is the
        completeness of genome ``i``.

        Returns
        -------
        tuple[pd.Series, dict[str, object]]
            Raw prevalence estimates indexed by gene family and any reusable
            fit artifacts. The unbiased estimator does not currently cache
            additional artifacts.
        """
        inverse_lambdas = 1 / self.lambdas
        weighted_data = self.data.multiply(inverse_lambdas, axis=1)
        prevalence = weighted_data.mean(axis=1)
        return prevalence, {}

    def _fit_mle(self):
        """
        Estimate gene family prevalence with the maximum likelihood estimator.

        For gene family ``j``, the Bernoulli likelihood is:

        .. math::

            L(\\pi_j \\mid X, \\lambda)
            = \\prod_{i=1}^{n}
            (\\pi_j \\lambda_i)^{X_{ij}}
            (1 - \\pi_j \\lambda_i)^{1 - X_{ij}}

        and the log-likelihood is:

        .. math::

            \\ell(\\pi_j \\mid X, \\lambda)
            = \\sum_{i=1}^{n}
            X_{ij} \\log(\\pi_j \\lambda_i)
            + (1 - X_{ij}) \\log(1 - \\pi_j \\lambda_i)

        This method minimizes the negative log-likelihood over
        ``0 <= pi_j <= 1``. To avoid undefined logarithms at ``0`` and ``1``,
        probabilities are clipped to ``[epsilon, 1 - epsilon]`` before taking
        logs.

        Returns
        -------
        tuple[pd.Series, dict[str, object]]
            Raw prevalence estimates indexed by gene family and cached
            per-gene objective values under ``nll_at_hat`` for reuse by
            likelihood-ratio confidence intervals.
        """
        mle_estimates = np.empty(self._data_np.shape[0], dtype=float)
        mle_objectives = np.empty(self._data_np.shape[0], dtype=float)
        for gene_index, data_j in enumerate(self._data_np):
            pi_hat, nll_at_hat = self._fit_mle_gene(data_j)
            mle_estimates[gene_index] = pi_hat
            mle_objectives[gene_index] = nll_at_hat

        return pd.Series(mle_estimates, index=self.data.index), {
            "nll_at_hat": pd.Series(mle_objectives, index=self.data.index),
        }

    def negative_log_likelihood(self, pi_j, data_j):
        """
        Evaluate the Bernoulli negative log-likelihood for one gene family.

        For gene family ``j``, the negative log-likelihood is:

        .. math::

            -\\ell(\\pi_j \\mid X, \\lambda)
            = -\\sum_{i=1}^{n}
            \\left[
            X_{ij} \\log(\\pi_j \\lambda_i)
            + (1 - X_{ij}) \\log(1 - \\pi_j \\lambda_i)
            \\right]

        where ``X_{ij}`` is the observed binary presence/absence value for
        gene family ``j`` in genome ``i`` and ``\\lambda_i`` is the
        completeness of genome ``i``.

        To keep the logarithms defined at the boundaries, the probabilities
        ``pi_j * lambda_i`` are clipped to
        ``[self._MLE_EPSILON, 1 - self._MLE_EPSILON]`` before evaluation.

        Parameters
        ----------
        pi_j : float
            Candidate prevalence value satisfying ``0 <= pi_j <= 1``.
        data_j : array-like
            Observed binary presence/absence values for one gene family across
            all genomes.

        Returns
        -------
        float
            Negative log-likelihood evaluated at ``pi_j``.
        """
        observations = np.asarray(data_j, dtype=float)
        probabilities = np.clip(
            pi_j * self._lambdas_np,
            self._MLE_EPSILON,
            1.0 - self._MLE_EPSILON,
        )
        log_likelihood = np.sum(
            observations * np.log(probabilities)
            + (1.0 - observations) * np.log1p(-probabilities)
        )
        return float(-log_likelihood)

    def _fit_mle_gene(self, data_j):
        """Fit the MLE and objective value for one gene family."""
        result = minimize_scalar(
            fun=lambda pi_j: self.negative_log_likelihood(pi_j, data_j),
            bounds=(0.0, 1.0),
            method="bounded",
            options={"maxiter": self._MLE_MAXITER},
        )

        # allow for boundary solutions by including 0 and 1 as candidates and taking the best
        candidate_estimates = np.array([0.0, float(result.x), 1.0], dtype=float)
        candidate_objectives = np.array(
            [self.negative_log_likelihood(candidate, data_j) for candidate in candidate_estimates]
        )
        best_index = int(np.argmin(candidate_objectives))
        return float(candidate_estimates[best_index]), float(candidate_objectives[best_index])

    def _estimate_interval(self, estimator, interval_method, prevalence, alpha, fit_metadata):
        """
        Dispatch to the requested confidence interval method.

        Returns
        -------
        ConfidenceIntervalResult
            Structured confidence interval result for one method/alpha pair.
        """
        if interval_method == "chebyshev":
            ci_lower, ci_upper, metadata = self._ci_chebyshev(
                estimator=estimator,
                prevalence=prevalence,
                alpha=alpha,
            )
            return ConfidenceIntervalResult(
                method=interval_method,
                alpha=alpha,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                metadata=metadata,
            )

        if interval_method == "likelihood_ratio":
            ci_lower, ci_upper, metadata = self._ci_likelihood_ratio(
                estimator=estimator,
                prevalence=prevalence,
                alpha=alpha,
                fit_metadata=fit_metadata,
            )
            return ConfidenceIntervalResult(
                method=interval_method,
                alpha=alpha,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                metadata=metadata,
            )

        raise ValueError(f"Unsupported interval_method {interval_method!r}.")

    def _estimate_intervals(self, estimator, interval_specs, prevalence, fit_metadata):
        """Estimate multiple confidence intervals, preserving request order."""
        if not interval_specs:
            return []

        if (
            estimator == "mle"
            and all(interval_spec["method"] == "likelihood_ratio" for interval_spec in interval_specs)
            and len(interval_specs) > 1
        ):
            return self._ci_likelihood_ratio_multiple(
                estimator=estimator,
                prevalence=prevalence,
                alphas=[float(interval_spec["alpha"]) for interval_spec in interval_specs],
                fit_metadata=fit_metadata,
            )

        return [
            self._estimate_interval(
                estimator=estimator,
                interval_method=interval_spec["method"],
                prevalence=prevalence,
                alpha=interval_spec["alpha"],
                fit_metadata=fit_metadata,
            )
            for interval_spec in interval_specs
        ]

    def _ci_chebyshev(self, estimator, prevalence, alpha):
        """
        Construct Chebyshev confidence intervals for the unbiased estimator.

        For gene family ``j``, the variance estimator is:

        .. math::

            \\mathrm{Var}(\\hat{\\pi}_j)
            = n^{-2} \\sum_{i=1}^{n}
            \\left(1 - \\hat{\\pi}_j \\lambda_i\\right) \\frac{\\hat{\\pi}_j}{\\lambda_i}

        where:

        - ``\\hat{\\pi}_j`` is the estimated prevalence for gene family ``j``
        - ``\\lambda_i`` is the completeness of genome ``i``
        - ``n`` is the number of genomes

        The Chebyshev interval width is then:

        .. math::

            \\mathrm{CI}_j = \\sqrt{\\frac{\\mathrm{Var}(\\hat{\\pi}_j)}{\\alpha}}

        and the confidence interval is
        ``[\\hat{\\pi}_j - \\mathrm{CI}_j, \\hat{\\pi}_j + \\mathrm{CI}_j]``,
        with lower bounds floored at ``0``.

        Parameters
        ----------
        estimator : {"unbiased"}
            Estimator associated with ``prevalence``. This argument is checked
            defensively to ensure the method is used only with the unbiased
            estimator.
        prevalence : pd.Series
            Raw prevalence estimates indexed by gene family.
        alpha : float
            Significance level satisfying ``0 < alpha < 1``.

        Returns
        -------
        tuple[pd.Series, pd.Series, dict[str, object]]
            Lower bounds, upper bounds, and interval metadata. The metadata
            dictionary contains the variance estimates under the ``variance``
            key.
        """
        if estimator != "unbiased":
            raise ValueError("The 'chebyshev' interval method requires estimator='unbiased'.")

        variance = self._estimate_unbiased_variance(prevalence=prevalence)
        ci_widths = np.sqrt(variance / alpha)
        lower_bounds = (prevalence - ci_widths).clip(lower=0)
        upper_bounds = prevalence + ci_widths
        metadata = {"variance": variance}
        return lower_bounds, upper_bounds, metadata

    def _estimate_unbiased_variance(self, prevalence):
        """
        Estimate the variance of the unbiased prevalence estimator.

        Parameters
        ----------
        prevalence : pd.Series
            Raw unbiased prevalence estimates indexed by gene family.

        Returns
        -------
        pd.Series
            Variance estimates indexed by gene family.
        """
        n = len(self.lambdas)
        scaling_factor = 1 / (n**2)
        pi_lambda_matrix = np.outer(prevalence, 1 / self.lambdas) * (
            1 - np.outer(prevalence, self.lambdas)
        )
        variance = scaling_factor * pi_lambda_matrix.sum(axis=1)
        return pd.Series(variance, index=prevalence.index)

    def _ci_likelihood_ratio(self, estimator, prevalence, alpha, fit_metadata):
        """
        Construct likelihood-ratio confidence intervals for the MLE.

        For gene family ``j``, the confidence interval is the set of values
        :math:`\\pi_j` satisfying:

        .. math::

            2 \\left\\{
            \\mathrm{NLL}(\\pi_j)
            - \\mathrm{NLL}(\\hat{\\pi}^{\\mathrm{MLE}}_j)
            \\right\\}
            \\leq \\chi^2_{1, 1 - \\alpha}

        where :math:`\\mathrm{NLL}(\\pi_j) = -\\ell(\\pi_j \\mid X, \\lambda)`
        is the negative log-likelihood and
        :math:`\\chi^2_{1, 1 - \\alpha}` is the :math:`1 - \\alpha` quantile
        of a chi-square distribution with one degree of freedom.

        The interval endpoints are obtained by root finding on

        .. math::

            g(\\pi_j)
            =
            2 \\left\\{
            \\mathrm{NLL}(\\pi_j)
            - \\mathrm{NLL}(\\hat{\\pi}^{\\mathrm{MLE}}_j)
            \\right\\}
            - \\chi^2_{1, 1 - \\alpha}

        using Brent's method on each side of the MLE.

        Parameters
        ----------
        estimator : {"mle"}
            Estimator associated with ``prevalence``. This argument is checked
            defensively to ensure the method is used only with the MLE.
        prevalence : pd.Series
            Raw MLE prevalence estimates indexed by gene family.
        alpha : float
            Significance level satisfying ``0 < alpha < 1``.
        fit_metadata : dict[str, object]
            Cached artifacts from the MLE fit. This must include per-gene
            negative log-likelihood values at the fitted MLE under the
            ``nll_at_hat`` key.

        Returns
        -------
        tuple[pd.Series, pd.Series, dict[str, object]]
            Lower bounds, upper bounds, and interval metadata. The metadata
            dictionary contains the chi-square cutoff under ``cutoff`` and the
            root-finding method name under ``solver``.
        """
        if estimator != "mle":
            raise ValueError("The 'likelihood_ratio' interval method requires estimator='mle'.")

        nll_at_hat = fit_metadata.get("nll_at_hat")
        if not isinstance(nll_at_hat, pd.Series):
            raise ValueError("Likelihood-ratio intervals require cached MLE fit metadata.")
        cutoff = float(chi2.ppf(1.0 - alpha, df=1))
        prevalence_np = prevalence.to_numpy(dtype=float)
        nll_at_hat_np = nll_at_hat.to_numpy(dtype=float)
        lower_bounds = np.empty(self._data_np.shape[0], dtype=float)
        upper_bounds = np.empty(self._data_np.shape[0], dtype=float)

        for gene_index, data_j in enumerate(self._data_np):
            pi_hat = prevalence_np[gene_index]
            nll_at_hat_j = nll_at_hat_np[gene_index]

            def lr_gap(pi_j):
                return 2.0 * (self.negative_log_likelihood(pi_j, data_j) - nll_at_hat_j) - cutoff

            if pi_hat == 0.0 or lr_gap(0.0) <= 0.0:
                lower_bounds[gene_index] = 0.0
            else:
                lower_result = root_scalar(
                    lr_gap,
                    bracket=(0.0, pi_hat),
                    method=self._LR_SOLVER_METHOD,
                )
                lower_bounds[gene_index] = float(lower_result.root)

            if pi_hat == 1.0 or lr_gap(1.0) <= 0.0:
                upper_bounds[gene_index] = 1.0
            else:
                upper_result = root_scalar(
                    lr_gap,
                    bracket=(pi_hat, 1.0),
                    method=self._LR_SOLVER_METHOD,
                )
                upper_bounds[gene_index] = float(upper_result.root)

        metadata = {"cutoff": cutoff, "solver": f"root_scalar_{self._LR_SOLVER_METHOD}"}
        return (
            pd.Series(lower_bounds, index=prevalence.index),
            pd.Series(upper_bounds, index=prevalence.index),
            metadata,
        )

    def _ci_likelihood_ratio_multiple(self, estimator, prevalence, alphas, fit_metadata):
        """Construct likelihood-ratio confidence intervals for multiple alpha values."""
        if estimator != "mle":
            raise ValueError("The 'likelihood_ratio' interval method requires estimator='mle'.")

        nll_at_hat = fit_metadata.get("nll_at_hat")
        if not isinstance(nll_at_hat, pd.Series):
            raise ValueError("Likelihood-ratio intervals require cached MLE fit metadata.")

        prevalence_np = prevalence.to_numpy(dtype=float)
        nll_at_hat_np = nll_at_hat.to_numpy(dtype=float)
        unique_alphas = sorted({float(alpha) for alpha in alphas}, reverse=True)
        cutoffs = {alpha: float(chi2.ppf(1.0 - alpha, df=1)) for alpha in unique_alphas}
        lower_by_alpha = {alpha: np.empty(self._data_np.shape[0], dtype=float) for alpha in unique_alphas}
        upper_by_alpha = {alpha: np.empty(self._data_np.shape[0], dtype=float) for alpha in unique_alphas}

        for gene_index, data_j in enumerate(self._data_np):
            pi_hat = prevalence_np[gene_index]
            nll_at_hat_j = nll_at_hat_np[gene_index]

            def lr_gap(pi_j, cutoff):
                return 2.0 * (self.negative_log_likelihood(pi_j, data_j) - nll_at_hat_j) - cutoff

            previous_lower = pi_hat
            previous_upper = pi_hat

            for alpha in unique_alphas:
                cutoff = cutoffs[alpha]

                if pi_hat == 0.0 or lr_gap(0.0, cutoff) <= 0.0:
                    lower_root = 0.0
                else:
                    lower_bracket_upper = previous_lower
                    lower_result = root_scalar(
                        lambda pi_j, cutoff=cutoff: lr_gap(pi_j, cutoff),
                        bracket=(0.0, lower_bracket_upper),
                        method=self._LR_SOLVER_METHOD,
                    )
                    lower_root = float(lower_result.root)

                if pi_hat == 1.0 or lr_gap(1.0, cutoff) <= 0.0:
                    upper_root = 1.0
                else:
                    upper_bracket_lower = previous_upper
                    upper_result = root_scalar(
                        lambda pi_j, cutoff=cutoff: lr_gap(pi_j, cutoff),
                        bracket=(upper_bracket_lower, 1.0),
                        method=self._LR_SOLVER_METHOD,
                    )
                    upper_root = float(upper_result.root)

                lower_by_alpha[alpha][gene_index] = lower_root
                upper_by_alpha[alpha][gene_index] = upper_root
                previous_lower = lower_root
                previous_upper = upper_root

        intervals_by_alpha = {
            alpha: ConfidenceIntervalResult(
                method="likelihood_ratio",
                alpha=alpha,
                ci_lower=pd.Series(lower_by_alpha[alpha], index=prevalence.index),
                ci_upper=pd.Series(upper_by_alpha[alpha], index=prevalence.index),
                metadata={
                    "cutoff": cutoffs[alpha],
                    "solver": f"root_scalar_{self._LR_SOLVER_METHOD}",
                },
            )
            for alpha in unique_alphas
        }
        return [intervals_by_alpha[float(alpha)] for alpha in alphas]

    def _fit_with_interval_specs(
        self,
        *,
        estimator: str,
        intervals: list[dict[str, Any]],
        clip: bool,
        preserve_single_interval_metadata: bool,
    ) -> PrevalenceFitResult:
        """Compute one prevalence fit and zero or more confidence intervals."""
        prevalence, fit_metadata = self._estimate_prevalence(estimator=estimator)
        confidence_intervals = self._estimate_intervals(
            estimator=estimator,
            interval_specs=intervals,
            prevalence=prevalence,
            fit_metadata=fit_metadata,
        )

        if clip:
            prevalence, confidence_intervals = self._clip_fit_outputs(
                prevalence=prevalence,
                confidence_intervals=confidence_intervals,
            )

        interval_method = None
        alpha = None
        ci_lower = None
        ci_upper = None
        metadata: dict[str, object] = {}

        if len(confidence_intervals) == 1:
            interval = confidence_intervals[0]
            interval_method = interval.method
            alpha = interval.alpha
            ci_lower = interval.ci_lower
            ci_upper = interval.ci_upper
            if preserve_single_interval_metadata:
                metadata.update(interval.metadata)

        return PrevalenceFitResult(
            prevalence=prevalence,
            estimator=estimator,
            interval_method=interval_method,
            alpha=alpha,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            metadata=metadata,
            confidence_intervals=confidence_intervals,
        )

    def _clip_outputs(self, prevalence, ci_lower, ci_upper):
        """
        Clip fitted outputs to the valid prevalence range when requested.

        Parameters
        ----------
        prevalence : pd.Series
            Raw prevalence estimates indexed by gene family.
        ci_lower : pd.Series or None
            Lower confidence bounds, if available.
        ci_upper : pd.Series or None
            Upper confidence bounds, if available.

        Returns
        -------
        tuple[pd.Series, pd.Series or None, pd.Series or None]
            Clipped prevalence estimates and, when present, clipped interval
            bounds.
        """
        clipped_prevalence = prevalence.clip(upper=1.0)

        if ci_lower is None or ci_upper is None:
            return clipped_prevalence, ci_lower, ci_upper

        clipped_upper = ci_upper.clip(upper=1.0)
        clipped_lower = ci_lower.where(ci_lower <= clipped_prevalence, clipped_prevalence)
        return clipped_prevalence, clipped_lower, clipped_upper

    def _clip_fit_outputs(self, prevalence, confidence_intervals):
        """Clip a prevalence fit and its interval results to the valid range."""
        clipped_prevalence = prevalence.clip(upper=1.0)
        clipped_intervals = []

        for interval in confidence_intervals:
            clipped_lower, clipped_upper = self._clip_outputs(
                prevalence=clipped_prevalence,
                ci_lower=interval.ci_lower,
                ci_upper=interval.ci_upper,
            )[1:]
            clipped_intervals.append(
                ConfidenceIntervalResult(
                    method=interval.method,
                    alpha=interval.alpha,
                    ci_lower=clipped_lower,
                    ci_upper=clipped_upper,
                    metadata=interval.metadata,
                )
            )

        return clipped_prevalence, clipped_intervals
