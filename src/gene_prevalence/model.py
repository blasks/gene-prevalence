"""Bernoulli prevalence estimation model."""

import numpy as np
import pandas as pd


class BernoulliPrevalenceModel:
    """
    Estimate gene family prevalence with a Bernoulli model.

    This model assumes an observed binary presence/absence matrix ``data``,
    where rows correspond to gene families and columns correspond to genomes.
    Each genome has an associated completeness parameter ``lambda_i`` in
    ``lambdas``.

    The prevalence estimator for gene family ``j`` is:

    .. math::

        \\hat{\\pi}_j = \\frac{1}{n} \\sum_{i=1}^{n} \\frac{X_{ij}}{\\lambda_i}

    where:

    - ``X_{ij}`` is the observed presence/absence value for gene family ``j``
      in genome ``i``
    - ``\\lambda_i`` is the completeness of genome ``i``
    - ``n`` is the number of genomes

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing presence/absence data for gene families across
        genomes.
    lambdas : pd.Series
        A series of lambda values corresponding to the completeness of each
        genome in the dataset, with values between 0 and 1.
    clip_max : float or None, default=1.0
        Maximum value to clip the estimated prevalence. If set to ``None``,
        no clipping is applied.

    Attributes
    ----------
    data : pd.DataFrame
        Binary presence/absence data indexed by gene family and genome.
    lambdas : pd.Series
        Genome completeness values aligned to ``data.columns``.
    clip_max : float or None
        Upper bound applied to prevalence estimates when requested.
    pi_estimates_ : pd.Series
        Estimated prevalence values, created after calling
        :meth:`estimate_prevalence`.
    variance_ : pd.Series
        Estimated variances, created after calling :meth:`estimate_variance`.
    """

    def __init__(self, data, lambdas, clip_max=1.0):
        self.data = data.astype(int)
        self.lambdas = lambdas
        self.clip_max = clip_max
        self._validate_inputs()

    @property
    def pi_estimates_(self):
        """
        Return stored prevalence estimates, clipping on access when requested.

        Raises
        ------
        AttributeError
            If prevalence estimates have not been computed or assigned.
        """
        if not hasattr(self, "_pi_estimates"):
            raise AttributeError("pi_estimates_ has not been computed yet.")

        if self.clip_max is None:
            return self._pi_estimates

        return self._pi_estimates.clip(upper=self.clip_max)

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

    def estimate_prevalence(self):
        """
        Estimate and store gene family prevalence values.

        The estimator is:

        .. math::

            \\hat{\\pi}_j = \\frac{1}{n} \\sum_{i=1}^{n} \\frac{1}{\\lambda_i} X_{ij}

        where ``X_{ij}`` is the observed binary presence/absence value for
        gene family ``j`` in genome ``i``, and ``\\lambda_i`` is the
        completeness of genome ``i``.

        Returns
        -------
        pd.Series
            A series of prevalence estimates indexed by gene family. Raw
            values are stored internally and returned through
            ``self.pi_estimates_`` with clipping applied on access.
        """
        inverse_lambdas = 1 / self.lambdas
        weighted_data = self.data.multiply(inverse_lambdas, axis=1)
        pi_estimates = weighted_data.mean(axis=1)

        self._pi_estimates = pi_estimates
        return self.pi_estimates_

    def estimate_variance(self):
        """
        Estimate and store the variance of each prevalence estimate.

        If prevalence estimates are not already available, this method first
        calls :meth:`estimate_prevalence`. Variance is computed from the raw
        stored estimates before any clipping is applied on access.

        The variance estimator is:

        .. math::

            \\mathrm{Var}(\\hat{\\pi}_j)
            = n^{-2} \\sum_{i=1}^{n}
            \\left(1 - \\hat{\\pi}_j \\lambda_i\\right) \\frac{\\hat{\\pi}_j}{\\lambda_i}

        where:

        - ``\\hat{\\pi}_j`` is the estimated prevalence for gene family ``j``
        - ``\\lambda_i`` is the completeness of genome ``i``
        - ``n`` is the number of genomes

        Returns
        -------
        pd.Series
            A series of variance estimates indexed by gene family. The same
            values are also stored on ``self.variance_``.
        """
        if not hasattr(self, "pi_estimates_"):
            self.estimate_prevalence()

        raw_pi_estimates = self._pi_estimates
        n = len(self.lambdas)
        scaling_factor = 1 / (n**2)
        pi_lambda_matrix = np.outer(raw_pi_estimates, 1 / self.lambdas) * (
            1 - np.outer(raw_pi_estimates, self.lambdas)
        )
        variance = scaling_factor * pi_lambda_matrix.sum(axis=1)

        self.variance_ = pd.Series(variance, index=raw_pi_estimates.index)
        return self.variance_

    def confidence_interval(self, alpha, return_ci_widths=False):
        """
        Return confidence interval bounds or widths for each prevalence estimate.

        If ``self.variance_`` is not already available, this method first
        calls :meth:`estimate_variance`.

        The confidence interval width for gene family ``j`` is:

        .. math::

            \\mathrm{CI}_j = \\sqrt{\\frac{\\mathrm{Var}(\\hat{\\pi}_j)}{\\alpha}}

        where:

        - ``\\mathrm{Var}(\\hat{\\pi}_j)`` is the estimated variance of the
          prevalence estimator for gene family ``j``
        - ``\\alpha`` is the significance level (e.g. for a 95% confidence interval ``\\alpha = 0.05``)

        Parameters
        ----------
        alpha : float
            Significance level used to scale the variance estimate. Must
            satisfy ``0 < alpha < 1``. For example, for a 95% confidence interval, 
            set ``alpha = 0.05``.
        return_ci_widths : bool, default=False
            If ``True``, return only the confidence interval widths. When
            ``False``, return lower and upper confidence bounds.

        Returns
        -------
        pd.Series or tuple[pd.Series, pd.Series]
            If ``return_ci_widths`` is ``True``, returns a series of
            confidence interval widths indexed by gene family. Otherwise,
            returns lower and upper bound series indexed by gene family.

        Raises
        ------
        ValueError
            If ``alpha`` is not strictly between 0 and 1.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1.")

        if not hasattr(self, "variance_"):
            self.estimate_variance()

        ci_widths = np.sqrt(self.variance_ / alpha)
        if return_ci_widths:
            return ci_widths

        raw_pi_estimates = self._pi_estimates
        lower_bounds = (raw_pi_estimates - ci_widths).clip(lower=0)
        upper_bounds = raw_pi_estimates + ci_widths

        if self.clip_max is not None:
            clipped_pi_estimates = self.pi_estimates_
            upper_bounds = upper_bounds.clip(upper=self.clip_max)
            lower_bounds = lower_bounds.where(
                lower_bounds <= clipped_pi_estimates, clipped_pi_estimates
            )

        return lower_bounds, upper_bounds
