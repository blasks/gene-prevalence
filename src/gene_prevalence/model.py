"""Bernoulli prevalence estimation model and fit results."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


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
    - :math:`\pi_j` is the true prevalence of gene family ``j``
    - :math:`\lambda_i` is the completeness of genome ``i``

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

    def __init__(self, data, lambdas):
        self.data = data.astype(int)
        self.lambdas = lambdas
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

        The point estimator and interval method are specified independently so
        the API can support estimator-specific interval methods today and
        shared interval methods in the future. At present:

        - ``estimator="unbiased"`` may be paired with
          ``interval_method="chebyshev"``
        - ``estimator="mle"`` may be paired with
          ``interval_method="likelihood_ratio"``

        The currently implemented point estimator is the unbiased estimator:

        .. math::

            \\hat{\\pi}_j = \\frac{1}{n} \\sum_{i=1}^{n} \\frac{X_{ij}}{\\lambda_i}

        where:

        - :math:`X_{ij}` is the observed binary presence/absence value for
          gene family ``j`` in genome ``i``
        - :math:`\lambda_i` is the completeness of genome ``i``
        - :math:`n` is the number of genomes

        The currently implemented interval method is the Chebyshev interval
        based on the unbiased estimator variance:

        .. math::

            \\mathrm{Var}(\\hat{\\pi}_j)
            = n^{-2} \\sum_{i=1}^{n}
            \\left(1 - \\hat{\\pi}_j \\lambda_i\\right) \\frac{\\hat{\\pi}_j}{\\lambda_i}

        and interval width:

        .. math::

            \\mathrm{CI}_j = \\sqrt{\\frac{\\mathrm{Var}(\\hat{\\pi}_j)}{\\alpha}}

        Parameters
        ----------
        estimator : {"unbiased", "mle"}
            Point estimator to use. Only ``"unbiased"`` is currently
            implemented.
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
            The prevalence estimates and any fitted interval bounds.

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

        prevalence = self._estimate_prevalence(estimator=estimator)
        metadata: dict[str, object] = {}
        ci_lower = None
        ci_upper = None

        if interval_method is not None:
            ci_lower, ci_upper, interval_metadata = self._estimate_interval(
                estimator=estimator,
                interval_method=interval_method,
                prevalence=prevalence,
                alpha=alpha,
            )
            metadata.update(interval_metadata)

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
            If the index of ``lambdas`` does not match ``data.columns`` or if
            ``data`` contains values other than 0 and 1.
        """
        if not self.lambdas.index.equals(self.data.columns):
            raise ValueError("The index of lambdas must match the columns of data.")

        if not self.data.map(lambda x: x in [0, 1]).all().all():
            raise ValueError("Data must be binary (0 and 1).")

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
        else:
            if alpha is None:
                raise ValueError("alpha must be provided when interval_method is not None.")
            if not 0 < alpha < 1:
                raise ValueError("alpha must be strictly between 0 and 1.")

            allowed_intervals = self._ALLOWED_INTERVALS[estimator]
            if interval_method not in allowed_intervals:
                raise ValueError(
                    f"interval_method={interval_method!r} is not supported for "
                    f"estimator={estimator!r}."
                )

    def _estimate_prevalence(self, estimator):
        """Dispatch to the requested prevalence estimator."""
        if estimator == "unbiased":
            return self._fit_unbiased()

        if estimator == "mle":
            raise NotImplementedError("The 'mle' estimator is not implemented yet.")

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
        pd.Series
            Raw prevalence estimates indexed by gene family.
        """
        inverse_lambdas = 1 / self.lambdas
        weighted_data = self.data.multiply(inverse_lambdas, axis=1)
        return weighted_data.mean(axis=1)

    def _estimate_interval(self, estimator, interval_method, prevalence, alpha):
        """Dispatch to the requested confidence interval method."""
        if interval_method == "chebyshev":
            return self._ci_chebyshev(
                estimator=estimator,
                prevalence=prevalence,
                alpha=alpha,
            )

        if interval_method == "likelihood_ratio":
            raise NotImplementedError(
                "The 'likelihood_ratio' interval method is not implemented yet."
            )

        raise ValueError(f"Unsupported interval_method {interval_method!r}.")

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
