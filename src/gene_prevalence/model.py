"""Bernoulli prevalence estimation model and fit results."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, root_scalar
from scipy.stats import chi2, norm


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
    interval_method : {"chebyshev", "wald", "likelihood_ratio"} or None
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
        Additional method-specific outputs. For example, the unbiased
        estimator with the Chebyshev interval stores the variance estimates
        under ``metadata["variance"]``.
    """

    prevalence: pd.Series
    estimator: str
    interval_method: str | None
    alpha: float | None
    ci_lower: pd.Series | None = None
    ci_upper: pd.Series | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def confidence_interval(self) -> tuple[pd.Series, pd.Series]:
        """
        Return the fitted confidence interval bounds.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            Lower and upper confidence bounds indexed by gene family.

        Raises
        ------
        ValueError
            If this fit was created without a confidence interval.
        """
        if self.ci_lower is None or self.ci_upper is None:
            raise ValueError("No confidence interval was computed for this fit.")

        return self.ci_lower, self.ci_upper

    def summary(self) -> pd.DataFrame:
        """
        Return a tabular summary of the fitted quantities.

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by gene family. The ``prevalence`` column is
            always present. If confidence intervals were computed, the
            ``ci_lower`` and ``ci_upper`` columns are included. If the fit
            metadata contains a series under the ``variance`` key, it is also
            included as a ``variance`` column.
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


@dataclass(frozen=True)
class CombinedPrevalenceFitResult(PrevalenceFitResult):
    """
    Store the output of combining several sub-population fits into one.

    Parameters
    ----------
    combination_method : {"approx_var_likelihood"}
        Name of the method used to combine sub-population prevalence
        estimates and confidence intervals into this result.
    """

    combination_method: str = field(kw_only=True)

    _SUPPORTED_COMBINATION_METHODS = frozenset({"approx_var_likelihood"})

    def __post_init__(self):
        if self.combination_method not in self._SUPPORTED_COMBINATION_METHODS:
            raise ValueError(
                f"Unsupported combination_method {self.combination_method!r}. "
                f"Choose from {sorted(self._SUPPORTED_COMBINATION_METHODS)}."
            )


def _validate_combine_inputs(results, proportions):
    """Validate arguments passed to :func:`combine_approx_var_likelihood`."""
    if len(results) == 0:
        raise ValueError("results must contain at least one PrevalenceFitResult.")

    if len(results) != len(proportions):
        raise ValueError("results and proportions must have the same length.")

    proportions_array = np.asarray(proportions, dtype=float)
    if not np.isfinite(proportions_array).all() or (proportions_array <= 0.0).any():
        raise ValueError("proportions must contain only positive, finite values.")

    if not np.isclose(proportions_array.sum(), 1.0):
        raise ValueError("proportions must sum to 1.")

    reference = results[0]
    for result in results:
        if result.estimator != "mle":
            raise ValueError(
                "All results must use estimator='mle' to combine likelihood-ratio intervals."
            )

        if result.interval_method != "likelihood_ratio":
            raise ValueError(
                "All results must use interval_method='likelihood_ratio' to combine "
                "likelihood-ratio intervals."
            )

        if result.ci_lower is None or result.ci_upper is None:
            raise ValueError("All results must include a fitted confidence interval.")

        if result.alpha != reference.alpha:
            raise ValueError("All results must share the same alpha.")

        if not result.prevalence.index.equals(reference.prevalence.index):
            raise ValueError("All results must share the same gene family index.")


def _clip_combined_outputs(prevalence, ci_lower, ci_upper):
    """
    Clip combined outputs to the valid prevalence range.

    Unlike :meth:`BernoulliPrevalenceModel._clip_outputs`, this also clips
    the lower confidence bound to ``0`` because the normal approximation
    used by :func:`combine_approx_var_likelihood` can otherwise push it
    negative.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        Clipped prevalence estimates and interval bounds.
    """
    clipped_prevalence = prevalence.clip(lower=0.0, upper=1.0)
    clipped_upper = ci_upper.clip(upper=1.0)
    clipped_lower = ci_lower.clip(lower=0.0)
    clipped_lower = clipped_lower.where(clipped_lower <= clipped_prevalence, clipped_prevalence)
    return clipped_prevalence, clipped_lower, clipped_upper


def combine_approx_var_likelihood(
    results: Sequence[PrevalenceFitResult],
    proportions: Sequence[float] | np.ndarray | pd.Series,
    *,
    clip: bool = True,
) -> CombinedPrevalenceFitResult:
    """
    Combine prevalence estimates and their corresponding likelihood-ratio 
    confidence intervals from several sub-populations into one combined estimate.

    Each sub-population's likelihood-ratio confidence interval is
    approximated as a normal interval of the form
    ``pi_hat +/- c_alpha * sqrt(variance)`` in order to back out an implied
    variance. Point estimates and variances are then combined as weighted
    sums using ``proportions``, and a new normal-style interval is built
    from the combined point estimate and variance.

    Parameters
    ----------
    results : sequence of PrevalenceFitResult
        Per-sub-population fits, each produced with ``estimator="mle"`` and
        ``interval_method="likelihood_ratio"``, sharing the same ``alpha``
        and gene family index.
    proportions : sequence of float
        Mixture weights ``p_k`` for each sub-population in ``results``.
        Must be positive and sum to ``1``.
    clip : bool
        If ``True``, clip the combined prevalence and interval bounds to
        ``[0, 1]``.

    Returns
    -------
    CombinedPrevalenceFitResult
        Combined prevalence estimate and confidence interval, with
        ``combination_method="approx_var_likelihood"``.
    """
    _validate_combine_inputs(results=results, proportions=proportions)

    gene_index = results[0].prevalence.index
    alpha = results[0].alpha
    proportions_array = np.asarray(proportions, dtype=float)

    pi_hats = np.stack([result.prevalence.to_numpy(dtype=float) for result in results])
    lowers = np.stack([result.ci_lower.to_numpy(dtype=float) for result in results])
    uppers = np.stack([result.ci_upper.to_numpy(dtype=float) for result in results])

    c_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    half_widths = (uppers - lowers) / 2.0
    var_hats = (half_widths / c_alpha) ** 2

    combined_pi = proportions_array @ pi_hats
    combined_var = (proportions_array**2) @ var_hats
    combined_lower = combined_pi - c_alpha * np.sqrt(combined_var)
    combined_upper = combined_pi + c_alpha * np.sqrt(combined_var)

    prevalence = pd.Series(combined_pi, index=gene_index)
    ci_lower = pd.Series(combined_lower, index=gene_index)
    ci_upper = pd.Series(combined_upper, index=gene_index)
    variance = pd.Series(combined_var, index=gene_index)

    if clip:
        prevalence, ci_lower, ci_upper = _clip_combined_outputs(
            prevalence=prevalence,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    return CombinedPrevalenceFitResult(
        prevalence=prevalence,
        estimator="mle",
        interval_method="likelihood_ratio",
        alpha=alpha,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        combination_method="approx_var_likelihood",
        metadata={
            "proportions": proportions_array,
            "variance": variance,
            "c_alpha": c_alpha,
        },
    )


class BernoulliPrevalenceModel:
    """
    Estimate gene family prevalence with a Bernoulli observation model.

    This model assumes an observed binary presence/absence matrix ``data``,
    where rows correspond to genomes and columns correspond to gene families.
    Each genome has an associated completeness parameter ``lambda_i`` in
    ``lambdas``. Optionally, each gene family may also have a mean copy
    number ``c_j`` in ``copy_numbers``. Copy numbers are only used by the MLE
    path.

    The observation model is

    .. math::

        X_{ij} \\sim \\mathrm{Bernoulli}(\\pi_j \\lambda_i)

    where ``pi_j`` is the underlying prevalence of gene family ``j`` and
    ``lambda_i`` is the completeness of genome ``i``.

    When ``copy_numbers`` are provided, the MLE path replaces ``lambda_i`` for
    gene family ``j`` with the adjusted detection probability

    .. math::

        \\lambda_{ij}^{\\mathrm{adj}} = 1 - (1 - \\lambda_i)^{c_j}
    """

    _SUPPORTED_ESTIMATORS = {"unbiased", "mle"}
    _SUPPORTED_INTERVAL_METHODS = {"chebyshev", "wald", "likelihood_ratio"}
    _ALLOWED_INTERVALS = {
        "unbiased": {"chebyshev", "wald"},
        "mle": {"likelihood_ratio"},
    }
    _MLE_EPSILON = 1e-8
    _MLE_MAXITER = 500
    _LR_SOLVER_METHOD = "brentq"

    def __init__(self, data, lambdas, copy_numbers=None):
        self.data = data.astype(int)
        self.lambdas = lambdas
        self.copy_numbers = copy_numbers
        self._validate_inputs()

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

        Supported estimator/interval pairings are:

        - ``estimator="unbiased"`` with ``interval_method=None``,
          ``interval_method="chebyshev"``, or ``interval_method="wald"``
        - ``estimator="mle"`` with ``interval_method=None`` or
          ``interval_method="likelihood_ratio"``

        For the ``mle`` plus ``likelihood_ratio`` case, this method delegates
        directly to :meth:`_ci_likelihood_ratio` so the MLE prevalence and the
        likelihood-ratio interval are computed in one pass.

        Parameters
        ----------
        estimator : {"unbiased", "mle"}
            Point estimator to fit.
        interval_method : {"chebyshev", "wald", "likelihood_ratio"} or None
            Optional confidence interval method. Pass ``None`` to fit only the
            point estimator.
        alpha : float or None
            Significance level for the confidence interval. This must be
            ``None`` when ``interval_method`` is ``None`` and must satisfy
            ``0 < alpha < 1`` otherwise.
        clip : bool
            If ``True``, clip prevalence estimates and interval upper bounds
            to at most ``1``.

        Returns
        -------
        PrevalenceFitResult
            Fitted prevalence estimates and, when requested, one confidence
            interval.
        """
        self._validate_fit_arguments(
            estimator=estimator,
            interval_method=interval_method,
            alpha=alpha,
            clip=clip,
        )

        metadata: dict[str, object] = {}

        if estimator == "mle" and interval_method == "likelihood_ratio":
            prevalence, ci_lower, ci_upper, metadata = self._ci_likelihood_ratio(
                alpha=alpha,
                return_prevalence=True,
            )
        else:
            prevalence = self._estimate_prevalence(estimator=estimator)
            ci_lower = None
            ci_upper = None

            if interval_method is not None:
                ci_lower, ci_upper, metadata = self._estimate_interval(
                    estimator=estimator,
                    interval_method=interval_method,
                    prevalence=prevalence,
                    alpha=alpha,
                )

        if clip:
            prevalence, ci_lower, ci_upper = self._clip_outputs(
                prevalence=prevalence,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            )

        return PrevalenceFitResult(
            prevalence=prevalence,
            estimator=estimator,
            interval_method=interval_method,
            alpha=alpha,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            metadata=metadata,
        )

    def _validate_inputs(self):
        """
        Validate model inputs.

        Raises
        ------
        ValueError
            If ``lambdas`` are not aligned to ``data.index`` or if
            ``data`` contains values other than ``0`` and ``1``.
        """
        if not self.lambdas.index.equals(self.data.index):
            raise ValueError("The index of lambdas must match the rows of data.")

        if self.copy_numbers is not None:
            if not self.copy_numbers.index.equals(self.data.columns):
                raise ValueError("The index of copy_numbers must match the columns of data.")

            copy_number_values = self.copy_numbers.to_numpy(dtype=float)
            if not np.isfinite(copy_number_values).all():
                raise ValueError("copy_numbers must contain only finite values.")

            if (copy_number_values < 0.0).any():
                raise ValueError("copy_numbers must be non-negative.")

        if not self.data.map(lambda x: x in [0, 1]).all().all():
            raise ValueError("Data must be binary (0 and 1).")

    def _get_mle_lambdas(self, gene):
        """
        Return the MLE detection probabilities for one gene family.

        Parameters
        ----------
        gene : hashable
            Gene family label from ``data.columns``.

        Returns
        -------
        np.ndarray
            Per-genome detection probabilities used by the MLE path for the
            requested gene family.
        """
        lambdas = self.lambdas.to_numpy(dtype=float)
        if self.copy_numbers is None:
            return lambdas

        copy_number = float(self.copy_numbers.loc[gene])
        return 1.0 - np.power(1.0 - lambdas, copy_number)

    def _validate_fit_arguments(self, estimator, interval_method, alpha, clip):
        """Validate arguments passed to :meth:`fit`."""
        if estimator not in self._SUPPORTED_ESTIMATORS:
            raise ValueError(
                f"Unsupported estimator {estimator!r}. "
                f"Choose from {sorted(self._SUPPORTED_ESTIMATORS)}."
            )

        if interval_method is not None and interval_method not in self._SUPPORTED_INTERVAL_METHODS:
            raise ValueError(
                f"Unsupported interval_method {interval_method!r}. "
                f"Choose from {sorted(self._SUPPORTED_INTERVAL_METHODS)} or None."
            )

        if not isinstance(clip, bool):
            raise ValueError("clip must be a boolean value.")

        if interval_method is None:
            if alpha is not None:
                raise ValueError("alpha must be None when interval_method is None.")
            return

        if alpha is None:
            raise ValueError("alpha must be provided when interval_method is not None.")

        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1.")

        allowed_intervals = self._ALLOWED_INTERVALS[estimator]
        if interval_method not in allowed_intervals:
            raise ValueError(
                f"interval_method={interval_method!r} is not supported for estimator={estimator!r}."
            )

    def _estimate_prevalence(self, estimator):
        """
        Dispatch to the requested prevalence estimator.

        Returns
        -------
        pd.Series
            Prevalence estimates indexed by gene family.
        """
        if estimator == "unbiased":
            return self._fit_unbiased()

        if estimator == "mle":
            return self._fit_mle()

        raise ValueError(f"Unsupported estimator {estimator!r}.")

    def _fit_unbiased(self):
        """
        Estimate gene family prevalence with the unbiased estimator.

        Returns
        -------
        pd.Series
            Raw prevalence estimates indexed by gene family.
        """
        inverse_lambdas = 1 / self.lambdas
        weighted_data = self.data.multiply(inverse_lambdas, axis=0)
        return weighted_data.mean(axis=0)

    def _fit_mle(self):
        """
        Estimate gene family prevalence with the maximum likelihood estimator.

        Returns
        -------
        pd.Series
            Raw MLE prevalence estimates indexed by gene family.
        """
        data = self.data.to_numpy(dtype=float)
        mle_estimates = np.empty(data.shape[1], dtype=float)

        for gene_index, (gene, data_j) in enumerate(zip(self.data.columns, data.T, strict=False)):
            pi_hat, _ = self._fit_mle_gene(data_j, self._get_mle_lambdas(gene))
            mle_estimates[gene_index] = pi_hat

        return pd.Series(mle_estimates, index=self.data.columns)

    def negative_log_likelihood(self, pi_j, data_j, lambdas=None):
        """
        Evaluate the Bernoulli negative log-likelihood for one gene family.

        Parameters
        ----------
        pi_j : float
            Candidate prevalence satisfying ``0 <= pi_j <= 1``.
        data_j : array-like
            Presence/absence observations for one gene family across genomes.
        lambdas : array-like or None
            Per-genome detection probabilities for this gene family. When
            omitted, the raw model ``lambdas`` are used.

        Returns
        -------
        float
            Negative log-likelihood evaluated at ``pi_j``.
        """
        if lambdas is None:
            lambdas = self.lambdas.to_numpy(dtype=float)
        else:
            lambdas = np.asarray(lambdas, dtype=float)
        observations = np.asarray(data_j, dtype=float)
        probabilities = np.clip(pi_j * lambdas, self._MLE_EPSILON, 1.0 - self._MLE_EPSILON)
        log_likelihood = np.sum(
            observations * np.log(probabilities)
            + (1.0 - observations) * np.log1p(-probabilities)
        )
        return float(-log_likelihood)

    def _fit_mle_gene(self, data_j, lambdas):
        """
        Fit the MLE and objective value for one gene family.

        Returns
        -------
        tuple[float, float]
            The fitted prevalence and the corresponding negative
            log-likelihood value.
        """
        result = minimize_scalar(
            fun=lambda pi_j: self.negative_log_likelihood(pi_j, data_j, lambdas=lambdas),
            bounds=(0.0, 1.0),
            method="bounded",
            options={"maxiter": self._MLE_MAXITER},
        )

        candidate_estimates = np.array([0.0, float(result.x), 1.0], dtype=float)
        candidate_objectives = np.array(
            [
                self.negative_log_likelihood(candidate, data_j, lambdas=lambdas)
                for candidate in candidate_estimates
            ]
        )
        best_index = int(np.argmin(candidate_objectives))
        return float(candidate_estimates[best_index]), float(candidate_objectives[best_index])

    def _estimate_interval(self, estimator, interval_method, prevalence, alpha):
        """
        Dispatch to the requested confidence interval method.

        Parameters
        ----------
        estimator : {"unbiased", "mle"}
            Estimator associated with ``prevalence``.
        interval_method : {"chebyshev", "wald", "likelihood_ratio"}
            Confidence interval method to apply.
        prevalence : pd.Series
            Fitted prevalence estimates. For likelihood-ratio intervals this
            is passed through to :meth:`_ci_likelihood_ratio` so the returned
            interval bounds use the same index.
        alpha : float
            Significance level satisfying ``0 < alpha < 1``.

        Returns
        -------
        tuple[pd.Series, pd.Series, dict[str, object]]
            Lower bounds, upper bounds, and interval metadata.
        """
        if interval_method == "chebyshev":
            return self._ci_chebyshev(
                estimator=estimator,
                prevalence=prevalence,
                alpha=alpha,
            )

        if interval_method == "wald":
            return self._ci_wald(
                estimator=estimator,
                prevalence=prevalence,
                alpha=alpha,
            )

        if interval_method == "likelihood_ratio":
            return self._ci_likelihood_ratio(
                alpha=alpha,
                prevalence=prevalence,
                return_prevalence=False,
            )

        raise ValueError(f"Unsupported interval_method {interval_method!r}.")

    def _ci_chebyshev(self, estimator, prevalence, alpha):
        """
        Construct Chebyshev confidence intervals for the unbiased estimator.

        Returns
        -------
        tuple[pd.Series, pd.Series, dict[str, object]]
            Lower bounds, upper bounds, and metadata containing the variance
            estimate under ``"variance"``.
        """
        if estimator != "unbiased":
            raise ValueError("The 'chebyshev' interval method requires estimator='unbiased'.")

        variance = self._estimate_unbiased_variance(prevalence=prevalence)
        ci_widths = np.sqrt(variance / alpha)
        lower_bounds = (prevalence - ci_widths).clip(lower=0)
        upper_bounds = prevalence + ci_widths
        metadata = {"variance": variance}
        return lower_bounds, upper_bounds, metadata

    def _ci_wald(self, estimator, prevalence, alpha):
        """
        Construct Wald confidence intervals for the unbiased estimator.

        Returns
        -------
        tuple[pd.Series, pd.Series, dict[str, object]]
            Lower bounds, upper bounds, and metadata containing the variance
            estimate under ``"variance"``.
        """
        if estimator != "unbiased":
            raise ValueError("The 'wald' interval method requires estimator='unbiased'.")

        variance = self._estimate_unbiased_variance(prevalence=prevalence)
        ci_widths = np.sqrt(variance) * norm.ppf(1 - alpha / 2)
        lower_bounds = (prevalence - ci_widths).clip(lower=0)
        upper_bounds = prevalence + ci_widths
        metadata = {"variance": variance}
        return lower_bounds, upper_bounds, metadata

    def _estimate_unbiased_variance(self, prevalence):
        """
        Estimate the variance of the unbiased prevalence estimator.

        Returns
        -------
        pd.Series
            Variance estimates indexed by gene family.
        """
        n = len(self.lambdas)
        scaling_factor = 1 / (n**2)
        prevalence_values = prevalence.to_numpy(dtype=float)
        lambda_values = self.lambdas.to_numpy(dtype=float)
        pi_lambda_matrix = np.outer(prevalence_values, 1 / lambda_values) * (
            1 - np.outer(prevalence_values, lambda_values)
        )
        variance = scaling_factor * pi_lambda_matrix.sum(axis=1)
        return pd.Series(variance, index=prevalence.index)

    def _ci_likelihood_ratio(self, alpha, prevalence=None, return_prevalence=False):
        """
        Construct likelihood-ratio confidence intervals for the MLE.

        When ``return_prevalence`` is ``True``, this method also returns the
        fitted MLE prevalence so the MLE does not need to be computed twice.

        Parameters
        ----------
        alpha : float
            Significance level satisfying ``0 < alpha < 1``.
        prevalence : pd.Series or None
            Optional prevalence estimates to use for the returned index. When
            omitted, the prevalence is assembled from the MLEs computed inside
            this method.
        return_prevalence : bool
            If ``True``, also return the fitted MLE prevalence series.

        Returns
        -------
        tuple
            If ``return_prevalence`` is ``False``, returns
            ``(ci_lower, ci_upper, metadata)``. If ``True``, returns
            ``(prevalence, ci_lower, ci_upper, metadata)``.
        """
        data = self.data.to_numpy(dtype=float)
        cutoff = float(chi2.ppf(1.0 - alpha, df=1))
        lower_bounds = np.empty(data.shape[1], dtype=float)
        upper_bounds = np.empty(data.shape[1], dtype=float)
        mle_estimates = np.empty(data.shape[1], dtype=float)

        for gene_index, (gene, data_j) in enumerate(zip(self.data.columns, data.T, strict=False)):
            mle_lambdas = self._get_mle_lambdas(gene)
            pi_hat, nll_at_hat = self._fit_mle_gene(data_j, mle_lambdas)
            mle_estimates[gene_index] = pi_hat

            def lr_gap(pi_j):
                return (
                    2.0
                    * (self.negative_log_likelihood(pi_j, data_j, lambdas=mle_lambdas) - nll_at_hat)
                    - cutoff
                )

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

        if prevalence is None:
            prevalence = pd.Series(mle_estimates, index=self.data.columns)

        metadata = {"cutoff": cutoff, "solver": f"root_scalar_{self._LR_SOLVER_METHOD}"}
        ci_lower = pd.Series(lower_bounds, index=prevalence.index)
        ci_upper = pd.Series(upper_bounds, index=prevalence.index)

        if return_prevalence:
            return prevalence, ci_lower, ci_upper, metadata

        return ci_lower, ci_upper, metadata

    def _clip_outputs(self, prevalence, ci_lower, ci_upper):
        """
        Clip fitted outputs to the valid prevalence range when requested.

        Returns
        -------
        tuple[pd.Series, pd.Series | None, pd.Series | None]
            Clipped prevalence estimates and, when present, clipped interval
            bounds.
        """
        clipped_prevalence = prevalence.clip(upper=1.0)

        if ci_lower is None or ci_upper is None:
            return clipped_prevalence, ci_lower, ci_upper

        clipped_upper = ci_upper.clip(upper=1.0)
        clipped_lower = ci_lower.where(ci_lower <= clipped_prevalence, clipped_prevalence)
        return clipped_prevalence, clipped_lower, clipped_upper
