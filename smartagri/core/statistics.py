"""
Statistical Computing Module for SmartAgri

This module provides comprehensive statistical analysis capabilities
for agricultural and animal husbandry data, including:
- Descriptive statistics and data summarization
- Bayesian inference for uncertainty quantification
- Hypothesis testing for experimental analysis
- Time series analysis for temporal patterns
- Spatial statistics for field-level analysis

Applications:
    - Crop yield analysis and prediction intervals
    - Animal growth curve modeling
    - Treatment effect estimation in field trials
    - Seasonal pattern detection in sensor data
    - Spatial variability assessment in fields

Example:
    >>> from smartagri.core.statistics import DescriptiveStats, BayesianInference
    >>> # Analyze yield data
    >>> stats = DescriptiveStats()
    >>> summary = stats.describe(yield_data)
    >>> # Bayesian yield prediction
    >>> bayes = BayesianInference()
    >>> posterior = bayes.linear_regression(X, y, prior_mean=0, prior_var=10)
"""

import numpy as np
from typing import (
    Optional,
    Union,
    List,
    Tuple,
    Callable,
    Any,
    Dict,
    TypeVar,
)
from dataclasses import dataclass, field
from enum import Enum, auto
from scipy import stats as scipy_stats
from scipy import signal
import logging

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases
ArrayLike = Union[np.ndarray, List, Tuple]


@dataclass
class StatisticalSummary:
    """
    Container for descriptive statistics.

    Attributes:
        n: Sample size
        mean: Arithmetic mean
        std: Standard deviation
        var: Variance
        min: Minimum value
        max: Maximum value
        median: Median value
        q1: First quartile (25th percentile)
        q3: Third quartile (75th percentile)
        iqr: Interquartile range
        skewness: Skewness coefficient
        kurtosis: Kurtosis coefficient
        se: Standard error of mean
    """

    n: int
    mean: float
    std: float
    var: float
    min: float
    max: float
    median: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float
    se: float

    def confidence_interval(
        self,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for the mean.

        Args:
            confidence: Confidence level (default 0.95)

        Returns:
            Tuple of (lower, upper) bounds
        """
        alpha = 1 - confidence
        z = scipy_stats.norm.ppf(1 - alpha / 2)
        margin = z * self.se
        return (self.mean - margin, self.mean + margin)


@dataclass
class HypothesisTestResult:
    """
    Container for hypothesis test results.

    Attributes:
        statistic: Test statistic value
        p_value: P-value
        reject_null: Whether to reject null hypothesis at alpha level
        alpha: Significance level used
        effect_size: Effect size measure
        power: Statistical power (if calculated)
        ci: Confidence interval for difference
    """

    statistic: float
    p_value: float
    reject_null: bool
    alpha: float
    effect_size: Optional[float] = None
    power: Optional[float] = None
    ci: Optional[Tuple[float, float]] = None


class DescriptiveStats:
    """
    Comprehensive descriptive statistics for agricultural data.

    Provides robust methods for data summarization including
    outlier detection and missing data handling.

    Example:
        >>> stats = DescriptiveStats()
        >>> summary = stats.describe(crop_yields)
        >>> print(f"Mean yield: {summary.mean:.2f} +/- {summary.std:.2f}")
    """

    def __init__(self, ddof: int = 1):
        """
        Initialize descriptive statistics calculator.

        Args:
            ddof: Delta degrees of freedom for variance calculation
        """
        self.ddof = ddof

    def describe(
        self,
        data: ArrayLike,
        percentiles: Optional[List[float]] = None,
    ) -> StatisticalSummary:
        """
        Compute comprehensive descriptive statistics.

        Args:
            data: Input data array
            percentiles: Additional percentiles to compute

        Returns:
            StatisticalSummary object with all statistics
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]  # Remove NaN values

        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=self.ddof)
        var = np.var(data, ddof=self.ddof)

        q1, median, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1

        skewness = scipy_stats.skew(data)
        kurtosis = scipy_stats.kurtosis(data)
        se = std / np.sqrt(n)

        return StatisticalSummary(
            n=n,
            mean=mean,
            std=std,
            var=var,
            min=np.min(data),
            max=np.max(data),
            median=median,
            q1=q1,
            q3=q3,
            iqr=iqr,
            skewness=skewness,
            kurtosis=kurtosis,
            se=se,
        )

    def correlation_matrix(
        self,
        data: np.ndarray,
        method: str = "pearson",
    ) -> np.ndarray:
        """
        Compute correlation matrix.

        Args:
            data: 2D array with variables as columns
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Correlation matrix
        """
        if method == "pearson":
            return np.corrcoef(data, rowvar=False)
        elif method == "spearman":
            corr, _ = scipy_stats.spearmanr(data)
            return corr if data.shape[1] > 2 else np.array([[1, corr], [corr, 1]])
        elif method == "kendall":
            n_vars = data.shape[1]
            corr_matrix = np.eye(n_vars)
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    tau, _ = scipy_stats.kendalltau(data[:, i], data[:, j])
                    corr_matrix[i, j] = tau
                    corr_matrix[j, i] = tau
            return corr_matrix
        else:
            raise ValueError(f"Unknown correlation method: {method}")

    def covariance_matrix(
        self,
        data: np.ndarray,
        rowvar: bool = False,
    ) -> np.ndarray:
        """
        Compute covariance matrix.

        Args:
            data: 2D data array
            rowvar: If True, rows are variables

        Returns:
            Covariance matrix
        """
        return np.cov(data, rowvar=rowvar, ddof=self.ddof)

    def detect_outliers(
        self,
        data: ArrayLike,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> np.ndarray:
        """
        Detect outliers in data.

        Args:
            data: Input data
            method: Detection method ('iqr', 'zscore', 'mad')
            threshold: Threshold for outlier detection

        Returns:
            Boolean array indicating outliers
        """
        data = np.asarray(data).flatten()

        if method == "iqr":
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            return (data < lower) | (data > upper)

        elif method == "zscore":
            z_scores = np.abs(scipy_stats.zscore(data))
            return z_scores > threshold

        elif method == "mad":
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z = 0.6745 * (data - median) / mad
            return np.abs(modified_z) > threshold

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def moving_statistics(
        self,
        data: ArrayLike,
        window: int,
        statistic: str = "mean",
    ) -> np.ndarray:
        """
        Compute rolling window statistics.

        Args:
            data: Input time series
            window: Window size
            statistic: Statistic to compute ('mean', 'std', 'min', 'max', 'median')

        Returns:
            Array of rolling statistics
        """
        data = np.asarray(data)
        n = len(data)
        result = np.full(n, np.nan)

        stat_funcs = {
            "mean": np.mean,
            "std": np.std,
            "min": np.min,
            "max": np.max,
            "median": np.median,
        }

        func = stat_funcs.get(statistic, np.mean)

        for i in range(window - 1, n):
            window_data = data[i - window + 1:i + 1]
            result[i] = func(window_data)

        return result


class BayesianInference:
    """
    Bayesian statistical inference for agricultural modeling.

    Provides Bayesian methods for parameter estimation and
    uncertainty quantification in crop and livestock models.

    Features:
        - Conjugate prior updates
        - MCMC sampling for complex models
        - Bayesian linear regression
        - Credible interval computation

    Example:
        >>> bayes = BayesianInference()
        >>> # Estimate mean yield with uncertainty
        >>> posterior = bayes.normal_mean(
        ...     data=yields,
        ...     prior_mean=5000,
        ...     prior_var=1000000
        ... )
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize Bayesian inference engine.

        Args:
            seed: Random seed for reproducibility
        """
        self._rng = np.random.default_rng(seed)

    def normal_mean(
        self,
        data: ArrayLike,
        prior_mean: float,
        prior_var: float,
        known_var: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Bayesian inference for normal mean with conjugate prior.

        Args:
            data: Observed data
            prior_mean: Prior mean
            prior_var: Prior variance
            known_var: Known variance (estimated from data if None)

        Returns:
            Dict with posterior parameters
        """
        data = np.asarray(data).flatten()
        n = len(data)
        sample_mean = np.mean(data)

        if known_var is None:
            known_var = np.var(data, ddof=1)

        # Posterior precision (inverse variance)
        prior_prec = 1 / prior_var
        likelihood_prec = n / known_var

        posterior_prec = prior_prec + likelihood_prec
        posterior_var = 1 / posterior_prec

        # Posterior mean
        posterior_mean = posterior_var * (
            prior_prec * prior_mean + likelihood_prec * sample_mean
        )

        return {
            "posterior_mean": posterior_mean,
            "posterior_var": posterior_var,
            "posterior_std": np.sqrt(posterior_var),
            "credible_interval_95": (
                posterior_mean - 1.96 * np.sqrt(posterior_var),
                posterior_mean + 1.96 * np.sqrt(posterior_var),
            ),
        }

    def linear_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prior_mean: Optional[np.ndarray] = None,
        prior_cov: Optional[np.ndarray] = None,
        noise_var: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Bayesian linear regression with conjugate normal prior.

        Args:
            X: Design matrix (n_samples, n_features)
            y: Response variable (n_samples,)
            prior_mean: Prior mean for coefficients
            prior_cov: Prior covariance for coefficients
            noise_var: Known noise variance (estimated if None)

        Returns:
            Dict with posterior parameters and predictions
        """
        n, p = X.shape

        # Default prior (weakly informative)
        if prior_mean is None:
            prior_mean = np.zeros(p)
        if prior_cov is None:
            prior_cov = 100 * np.eye(p)

        # Estimate noise variance if not provided
        if noise_var is None:
            # Use OLS residual variance
            beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta_ols
            noise_var = np.var(residuals, ddof=p)

        # Posterior computation
        prior_prec = np.linalg.inv(prior_cov)
        posterior_prec = prior_prec + (X.T @ X) / noise_var
        posterior_cov = np.linalg.inv(posterior_prec)

        posterior_mean = posterior_cov @ (
            prior_prec @ prior_mean + (X.T @ y) / noise_var
        )

        return {
            "posterior_mean": posterior_mean,
            "posterior_cov": posterior_cov,
            "posterior_std": np.sqrt(np.diag(posterior_cov)),
            "noise_var": noise_var,
        }

    def credible_interval(
        self,
        samples: np.ndarray,
        level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Compute credible interval from posterior samples.

        Args:
            samples: Posterior samples
            level: Credibility level

        Returns:
            Tuple of (lower, upper) bounds
        """
        alpha = 1 - level
        lower = np.percentile(samples, 100 * alpha / 2)
        upper = np.percentile(samples, 100 * (1 - alpha / 2))
        return (lower, upper)

    def highest_density_interval(
        self,
        samples: np.ndarray,
        level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Compute highest density interval (HDI) from samples.

        The HDI is the shortest interval containing the specified
        probability mass.

        Args:
            samples: Posterior samples
            level: Credibility level

        Returns:
            Tuple of (lower, upper) bounds
        """
        samples = np.sort(samples)
        n = len(samples)

        # Number of samples in interval
        n_in_interval = int(np.ceil(level * n))

        # Find shortest interval
        min_width = np.inf
        best_interval = (samples[0], samples[n_in_interval - 1])

        for i in range(n - n_in_interval + 1):
            width = samples[i + n_in_interval - 1] - samples[i]
            if width < min_width:
                min_width = width
                best_interval = (samples[i], samples[i + n_in_interval - 1])

        return best_interval

    def mcmc_sample(
        self,
        log_posterior: Callable[[np.ndarray], float],
        initial: np.ndarray,
        n_samples: int = 1000,
        n_warmup: int = 500,
        step_size: float = 0.1,
    ) -> np.ndarray:
        """
        MCMC sampling using Metropolis-Hastings algorithm.

        Args:
            log_posterior: Function computing log posterior
            initial: Initial parameter values
            n_samples: Number of samples to generate
            n_warmup: Number of warmup samples to discard
            step_size: Proposal step size

        Returns:
            Array of posterior samples (n_samples, n_params)
        """
        n_params = len(initial)
        samples = np.zeros((n_samples + n_warmup, n_params))

        current = initial.copy()
        current_log_p = log_posterior(current)
        n_accepted = 0

        for i in range(n_samples + n_warmup):
            # Propose new state
            proposal = current + step_size * self._rng.normal(size=n_params)
            proposal_log_p = log_posterior(proposal)

            # Metropolis-Hastings acceptance
            log_alpha = proposal_log_p - current_log_p

            if np.log(self._rng.random()) < log_alpha:
                current = proposal
                current_log_p = proposal_log_p
                n_accepted += 1

            samples[i] = current

        acceptance_rate = n_accepted / (n_samples + n_warmup)
        logger.info(f"MCMC acceptance rate: {acceptance_rate:.2%}")

        # Return samples after warmup
        return samples[n_warmup:]


class HypothesisTesting:
    """
    Hypothesis testing for agricultural experiments.

    Provides classical and modern hypothesis tests for
    comparing treatments, detecting effects, and validating models.

    Example:
        >>> tester = HypothesisTesting(alpha=0.05)
        >>> result = tester.t_test(control_yields, treatment_yields)
        >>> if result.reject_null:
        ...     print(f"Significant difference (p={result.p_value:.4f})")
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize hypothesis testing.

        Args:
            alpha: Significance level
        """
        self.alpha = alpha

    def t_test(
        self,
        group1: ArrayLike,
        group2: Optional[ArrayLike] = None,
        mu: float = 0,
        alternative: str = "two-sided",
        paired: bool = False,
        equal_var: bool = True,
    ) -> HypothesisTestResult:
        """
        Perform t-test for means.

        Args:
            group1: First sample
            group2: Second sample (None for one-sample test)
            mu: Hypothesized mean (one-sample) or difference (two-sample)
            alternative: 'two-sided', 'less', or 'greater'
            paired: Paired samples test
            equal_var: Assume equal variances (two-sample)

        Returns:
            HypothesisTestResult with test results
        """
        group1 = np.asarray(group1).flatten()

        if group2 is None:
            # One-sample t-test
            statistic, p_value = scipy_stats.ttest_1samp(
                group1, mu, alternative=alternative
            )
            effect_size = (np.mean(group1) - mu) / np.std(group1, ddof=1)
        else:
            group2 = np.asarray(group2).flatten()

            if paired:
                statistic, p_value = scipy_stats.ttest_rel(
                    group1, group2, alternative=alternative
                )
                diff = group1 - group2
                effect_size = np.mean(diff) / np.std(diff, ddof=1)
            else:
                statistic, p_value = scipy_stats.ttest_ind(
                    group1, group2, equal_var=equal_var, alternative=alternative
                )
                # Cohen's d
                pooled_std = np.sqrt(
                    ((len(group1) - 1) * np.var(group1, ddof=1) +
                     (len(group2) - 1) * np.var(group2, ddof=1)) /
                    (len(group1) + len(group2) - 2)
                )
                effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std

        return HypothesisTestResult(
            statistic=statistic,
            p_value=p_value,
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=effect_size,
        )

    def anova(
        self,
        *groups: ArrayLike,
    ) -> HypothesisTestResult:
        """
        One-way ANOVA test.

        Args:
            *groups: Multiple groups to compare

        Returns:
            HypothesisTestResult with F-test results
        """
        statistic, p_value = scipy_stats.f_oneway(*groups)

        # Eta-squared effect size
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)

        ss_between = sum(
            len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups
        )
        ss_total = np.sum((all_data - grand_mean) ** 2)
        eta_squared = ss_between / ss_total

        return HypothesisTestResult(
            statistic=statistic,
            p_value=p_value,
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=eta_squared,
        )

    def chi_square(
        self,
        observed: ArrayLike,
        expected: Optional[ArrayLike] = None,
    ) -> HypothesisTestResult:
        """
        Chi-square goodness of fit test.

        Args:
            observed: Observed frequencies
            expected: Expected frequencies (uniform if None)

        Returns:
            HypothesisTestResult with chi-square results
        """
        observed = np.asarray(observed)

        if expected is None:
            expected = np.full_like(observed, np.sum(observed) / len(observed))
        else:
            expected = np.asarray(expected)

        statistic, p_value = scipy_stats.chisquare(observed, expected)

        return HypothesisTestResult(
            statistic=statistic,
            p_value=p_value,
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
        )

    def normality_test(
        self,
        data: ArrayLike,
        method: str = "shapiro",
    ) -> HypothesisTestResult:
        """
        Test for normality of data.

        Args:
            data: Input data
            method: Test method ('shapiro', 'anderson', 'dagostino')

        Returns:
            HypothesisTestResult
        """
        data = np.asarray(data).flatten()

        if method == "shapiro":
            statistic, p_value = scipy_stats.shapiro(data)
        elif method == "dagostino":
            statistic, p_value = scipy_stats.normaltest(data)
        else:
            raise ValueError(f"Unknown normality test: {method}")

        return HypothesisTestResult(
            statistic=statistic,
            p_value=p_value,
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
        )

    def mann_whitney(
        self,
        group1: ArrayLike,
        group2: ArrayLike,
        alternative: str = "two-sided",
    ) -> HypothesisTestResult:
        """
        Mann-Whitney U test (non-parametric alternative to t-test).

        Args:
            group1: First sample
            group2: Second sample
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            HypothesisTestResult
        """
        group1 = np.asarray(group1).flatten()
        group2 = np.asarray(group2).flatten()

        statistic, p_value = scipy_stats.mannwhitneyu(
            group1, group2, alternative=alternative
        )

        # Rank-biserial correlation as effect size
        n1, n2 = len(group1), len(group2)
        r = 1 - (2 * statistic) / (n1 * n2)

        return HypothesisTestResult(
            statistic=statistic,
            p_value=p_value,
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=r,
        )


class TimeSeriesAnalysis:
    """
    Time series analysis for agricultural temporal data.

    Provides methods for analyzing temporal patterns in:
    - Weather data
    - Crop growth measurements
    - Sensor readings
    - Market prices

    Example:
        >>> ts = TimeSeriesAnalysis()
        >>> # Decompose yield time series
        >>> trend, seasonal, residual = ts.decompose(yearly_yields)
        >>> # Detect change points
        >>> breaks = ts.change_point_detection(sensor_data)
    """

    def __init__(self):
        """Initialize time series analyzer."""
        pass

    def decompose(
        self,
        data: ArrayLike,
        period: Optional[int] = None,
        model: str = "additive",
    ) -> Dict[str, np.ndarray]:
        """
        Decompose time series into trend, seasonal, and residual.

        Args:
            data: Time series data
            period: Seasonal period (auto-detected if None)
            model: Decomposition model ('additive' or 'multiplicative')

        Returns:
            Dict with 'trend', 'seasonal', 'residual' components
        """
        data = np.asarray(data)

        # Auto-detect period if not provided
        if period is None:
            period = self._detect_period(data)

        n = len(data)

        # Compute trend using centered moving average
        if period % 2 == 0:
            # Even period requires two-pass average
            ma1 = np.convolve(data, np.ones(period) / period, mode="valid")
            trend = np.convolve(ma1, np.ones(2) / 2, mode="valid")
            pad = (n - len(trend)) // 2
            trend = np.pad(trend, (pad, n - len(trend) - pad), mode="edge")
        else:
            trend = np.convolve(
                data, np.ones(period) / period, mode="same"
            )

        # Detrend
        if model == "additive":
            detrended = data - trend
        else:  # multiplicative
            detrended = data / np.where(trend != 0, trend, 1)

        # Compute seasonal component
        seasonal = np.zeros_like(data)
        for i in range(period):
            indices = np.arange(i, n, period)
            seasonal[indices] = np.mean(detrended[indices])

        # Normalize seasonal component
        if model == "additive":
            seasonal = seasonal - np.mean(seasonal)
        else:
            seasonal = seasonal / np.mean(seasonal)

        # Compute residual
        if model == "additive":
            residual = data - trend - seasonal
        else:
            residual = data / (trend * seasonal)
            residual = np.where(np.isfinite(residual), residual, 1)

        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "period": period,
        }

    def _detect_period(self, data: np.ndarray) -> int:
        """Detect dominant period using autocorrelation."""
        n = len(data)
        acf = np.correlate(data - np.mean(data), data - np.mean(data), mode="full")
        acf = acf[n - 1:] / acf[n - 1]

        # Find first peak after lag 0
        for i in range(1, len(acf) - 1):
            if acf[i] > acf[i - 1] and acf[i] > acf[i + 1]:
                return i

        return 12  # Default to annual seasonality

    def autocorrelation(
        self,
        data: ArrayLike,
        max_lag: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute autocorrelation function.

        Args:
            data: Time series data
            max_lag: Maximum lag to compute

        Returns:
            Array of autocorrelation values
        """
        data = np.asarray(data)
        n = len(data)

        if max_lag is None:
            max_lag = min(n // 2, 40)

        data_centered = data - np.mean(data)
        acf = np.correlate(data_centered, data_centered, mode="full")
        acf = acf[n - 1:n + max_lag] / acf[n - 1]

        return acf

    def partial_autocorrelation(
        self,
        data: ArrayLike,
        max_lag: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute partial autocorrelation function.

        Args:
            data: Time series data
            max_lag: Maximum lag to compute

        Returns:
            Array of PACF values
        """
        data = np.asarray(data)
        n = len(data)

        if max_lag is None:
            max_lag = min(n // 2, 40)

        pacf = np.zeros(max_lag + 1)
        pacf[0] = 1.0

        # Levinson-Durbin recursion
        acf = self.autocorrelation(data, max_lag)
        phi = np.zeros((max_lag + 1, max_lag + 1))

        for k in range(1, max_lag + 1):
            num = acf[k] - sum(phi[k - 1, j] * acf[k - j] for j in range(1, k))
            den = 1 - sum(phi[k - 1, j] * acf[j] for j in range(1, k))

            phi[k, k] = num / den if den != 0 else 0
            pacf[k] = phi[k, k]

            for j in range(1, k):
                phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]

        return pacf

    def stationarity_test(
        self,
        data: ArrayLike,
        regression: str = "c",
    ) -> Dict[str, Any]:
        """
        Augmented Dickey-Fuller test for stationarity.

        Args:
            data: Time series data
            regression: Regression type ('c' constant, 'ct' trend, 'n' none)

        Returns:
            Dict with test statistic, p-value, and critical values
        """
        from scipy.stats import norm

        data = np.asarray(data)
        n = len(data)

        # Compute first difference
        diff = np.diff(data)

        # Simple ADF approximation
        y = diff[1:]
        x = data[:-2]

        if regression == "c":
            x = np.column_stack([np.ones(len(x)), x])
        elif regression == "ct":
            x = np.column_stack([np.ones(len(x)), np.arange(len(x)), x])
        else:
            x = x.reshape(-1, 1)

        # OLS estimation
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        residuals = y - x @ beta

        # Test statistic
        se = np.sqrt(np.var(residuals) / (x[:, -1] @ x[:, -1]))
        t_stat = beta[-1] / se

        # Approximate p-value (simplified)
        p_value = 2 * norm.cdf(t_stat) if t_stat < 0 else 2 * (1 - norm.cdf(t_stat))

        return {
            "statistic": t_stat,
            "p_value": p_value,
            "is_stationary": t_stat < -2.86,  # 5% critical value
        }

    def change_point_detection(
        self,
        data: ArrayLike,
        method: str = "cumsum",
        threshold: Optional[float] = None,
    ) -> List[int]:
        """
        Detect change points in time series.

        Args:
            data: Time series data
            method: Detection method ('cumsum', 'binary_segmentation')
            threshold: Detection threshold

        Returns:
            List of change point indices
        """
        data = np.asarray(data)
        n = len(data)

        if threshold is None:
            threshold = 2 * np.std(data)

        change_points = []

        if method == "cumsum":
            # CUSUM change point detection
            mean = np.mean(data)
            cumsum = np.cumsum(data - mean)

            # Find peaks in CUSUM
            for i in range(1, n - 1):
                if abs(cumsum[i] - cumsum[i - 1]) > threshold:
                    change_points.append(i)

        return change_points


class SpatialStatistics:
    """
    Spatial statistics for field-level agricultural analysis.

    Provides geostatistical methods for analyzing spatial
    patterns in agricultural data, including:
    - Variogram analysis
    - Kriging interpolation
    - Spatial autocorrelation
    - Hot spot analysis

    Example:
        >>> spatial = SpatialStatistics()
        >>> # Fit variogram to soil samples
        >>> variogram = spatial.empirical_variogram(coords, values)
        >>> # Interpolate to grid
        >>> grid_values = spatial.kriging(coords, values, grid_points)
    """

    def __init__(self):
        """Initialize spatial statistics analyzer."""
        pass

    def empirical_variogram(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        n_bins: int = 15,
        max_dist: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute empirical variogram.

        The variogram measures spatial autocorrelation as a
        function of distance.

        Args:
            coords: Point coordinates (n, 2)
            values: Values at each point (n,)
            n_bins: Number of distance bins
            max_dist: Maximum distance to consider

        Returns:
            Dict with 'distance', 'semivariance', 'n_pairs'
        """
        n = len(values)

        # Compute pairwise distances
        distances = []
        semivariances = []

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                semi = 0.5 * (values[i] - values[j]) ** 2
                distances.append(dist)
                semivariances.append(semi)

        distances = np.array(distances)
        semivariances = np.array(semivariances)

        if max_dist is None:
            max_dist = np.max(distances) / 2

        # Bin by distance
        bin_edges = np.linspace(0, max_dist, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        binned_semi = np.zeros(n_bins)
        n_pairs = np.zeros(n_bins, dtype=int)

        for i, (d, s) in enumerate(zip(distances, semivariances)):
            if d <= max_dist:
                bin_idx = min(int(d / max_dist * n_bins), n_bins - 1)
                binned_semi[bin_idx] += s
                n_pairs[bin_idx] += 1

        # Average within bins
        with np.errstate(divide="ignore", invalid="ignore"):
            binned_semi = np.where(n_pairs > 0, binned_semi / n_pairs, np.nan)

        return {
            "distance": bin_centers,
            "semivariance": binned_semi,
            "n_pairs": n_pairs,
        }

    def fit_variogram_model(
        self,
        variogram: Dict[str, np.ndarray],
        model: str = "spherical",
    ) -> Dict[str, float]:
        """
        Fit theoretical variogram model.

        Args:
            variogram: Empirical variogram dict
            model: Model type ('spherical', 'exponential', 'gaussian')

        Returns:
            Dict with model parameters (nugget, sill, range)
        """
        from scipy.optimize import curve_fit

        distances = variogram["distance"]
        semivar = variogram["semivariance"]

        # Remove NaN values
        mask = ~np.isnan(semivar)
        distances = distances[mask]
        semivar = semivar[mask]

        def spherical(h, nugget, sill, a):
            result = np.where(
                h <= a,
                nugget + sill * (1.5 * h / a - 0.5 * (h / a) ** 3),
                nugget + sill,
            )
            return result

        def exponential(h, nugget, sill, a):
            return nugget + sill * (1 - np.exp(-h / a))

        def gaussian(h, nugget, sill, a):
            return nugget + sill * (1 - np.exp(-(h / a) ** 2))

        models = {
            "spherical": spherical,
            "exponential": exponential,
            "gaussian": gaussian,
        }

        func = models.get(model, spherical)

        # Initial parameter guesses
        nugget_init = semivar[0] if len(semivar) > 0 else 0
        sill_init = np.max(semivar) - nugget_init
        range_init = distances[len(distances) // 2]

        try:
            params, _ = curve_fit(
                func,
                distances,
                semivar,
                p0=[nugget_init, sill_init, range_init],
                bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
            )

            return {
                "model": model,
                "nugget": params[0],
                "sill": params[1],
                "range": params[2],
            }
        except RuntimeError:
            logger.warning("Variogram fitting failed, using defaults")
            return {
                "model": model,
                "nugget": nugget_init,
                "sill": sill_init,
                "range": range_init,
            }

    def ordinary_kriging(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        target_coords: np.ndarray,
        variogram_params: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ordinary kriging interpolation.

        Args:
            coords: Known point coordinates (n, 2)
            values: Known values (n,)
            target_coords: Target coordinates (m, 2)
            variogram_params: Variogram model parameters

        Returns:
            Tuple of (predictions, variances)
        """
        n = len(values)
        m = len(target_coords)

        nugget = variogram_params["nugget"]
        sill = variogram_params["sill"]
        a = variogram_params["range"]
        model = variogram_params.get("model", "spherical")

        def covariance(h):
            """Compute covariance from variogram."""
            if model == "spherical":
                gamma = np.where(
                    h <= a,
                    nugget + sill * (1.5 * h / a - 0.5 * (h / a) ** 3),
                    nugget + sill,
                )
            elif model == "exponential":
                gamma = nugget + sill * (1 - np.exp(-h / a))
            else:  # gaussian
                gamma = nugget + sill * (1 - np.exp(-(h / a) ** 2))

            return (nugget + sill) - gamma

        # Build kriging system
        K = np.zeros((n + 1, n + 1))

        for i in range(n):
            for j in range(n):
                h = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                K[i, j] = covariance(h)

        K[:n, n] = 1
        K[n, :n] = 1
        K[n, n] = 0

        predictions = np.zeros(m)
        variances = np.zeros(m)

        for t in range(m):
            # Build RHS
            k = np.zeros(n + 1)
            for i in range(n):
                h = np.sqrt(np.sum((coords[i] - target_coords[t]) ** 2))
                k[i] = covariance(h)
            k[n] = 1

            # Solve kriging system
            try:
                weights = np.linalg.solve(K, k)
            except np.linalg.LinAlgError:
                weights = np.linalg.lstsq(K, k, rcond=None)[0]

            # Prediction
            predictions[t] = np.dot(weights[:n], values)

            # Variance
            variances[t] = (nugget + sill) - np.dot(weights, k)
            variances[t] = max(0, variances[t])

        return predictions, variances

    def morans_i(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        weight_type: str = "distance",
        bandwidth: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute Moran's I spatial autocorrelation.

        Args:
            coords: Point coordinates (n, 2)
            values: Values at points (n,)
            weight_type: Spatial weight type ('distance', 'binary')
            bandwidth: Distance threshold for weights

        Returns:
            Dict with Moran's I, expected I, and z-score
        """
        n = len(values)
        mean = np.mean(values)
        deviations = values - mean

        if bandwidth is None:
            all_dists = []
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                    all_dists.append(d)
            bandwidth = np.median(all_dists)

        # Compute spatial weights
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                    if weight_type == "distance":
                        W[i, j] = 1 / max(dist, 1e-10)
                    else:  # binary
                        W[i, j] = 1 if dist <= bandwidth else 0

        # Row standardize
        row_sums = W.sum(axis=1, keepdims=True)
        W = np.where(row_sums > 0, W / row_sums, 0)

        # Compute Moran's I
        numerator = n * np.sum(W * np.outer(deviations, deviations))
        denominator = np.sum(W) * np.sum(deviations ** 2)

        I = numerator / denominator if denominator != 0 else 0

        # Expected value and variance
        E_I = -1 / (n - 1)

        # Simplified variance approximation
        S0 = np.sum(W)
        S1 = 0.5 * np.sum((W + W.T) ** 2)
        S2 = np.sum((W.sum(axis=0) + W.sum(axis=1)) ** 2)

        n_sq = n ** 2
        var_I = (n * ((n_sq - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2) -
                 (n - 1) * ((n_sq - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)) / \
                ((n - 1) * (n - 2) * (n - 3) * S0 ** 2 + 1e-10)
        var_I = max(var_I - E_I ** 2, 1e-10)

        z = (I - E_I) / np.sqrt(var_I)

        return {
            "morans_i": I,
            "expected_i": E_I,
            "z_score": z,
            "p_value": 2 * (1 - scipy_stats.norm.cdf(abs(z))),
        }
