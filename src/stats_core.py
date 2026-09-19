"""The statistics this study needs, written against numpy alone.

Why this module exists rather than a call to statsmodels.

The analysis originally used statsmodels for the regressions and scipy for the
correlation and the t tests. Both are excellent libraries. Both are also
compiled extensions built against a particular version of numpy, and when the
deployed dashboard installed a slightly different combination of versions than
the one it was developed with, importing them failed and the whole page went
down with a redacted error.

Everything this study actually asks for is ordinary linear algebra:

  a least squares fit
  heteroskedasticity robust (HC3) standard errors
  cluster robust standard errors
  a Pearson correlation with a confidence interval
  a Welch two sample t test
  the Student t distribution, to turn all of that into intervals and p values

That is a few hundred lines of numpy and arithmetic. Writing it here means the
dashboard depends only on numpy and pandas, which removes the failure entirely
and makes the app start faster.

Every function here is checked against statsmodels and scipy in
tests/test_stats_core.py, which is how the rewrite was verified rather than
assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# The Student t distribution
# ---------------------------------------------------------------------------
#
# Only two things are needed: the two sided tail probability for a given t, and
# the t value that puts a given probability in the two tails. Both come from
# the regularized incomplete beta function, which is the standard route.


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """The continued fraction used to evaluate the incomplete beta function.

    This is the modified Lentz algorithm. It is the textbook method and it
    converges quickly for the range of values a t distribution produces.
    """
    tiny = 1.0e-30
    max_iterations = 300
    tolerance = 3.0e-14

    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d

    for m in range(1, max_iterations + 1):
        m2 = 2 * m

        # Even step.
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        # Odd step.
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta

        if abs(delta - 1.0) < tolerance:
            break

    return result


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b), the regularized incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    front = math.exp(
        a * math.log(x) + b * math.log(1.0 - x) - _log_beta(a, b)
    )
    # The continued fraction converges fast only on one side of the symmetry
    # point, so the other side is reached through the reflection identity.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - math.exp(
        b * math.log(1.0 - x) + a * math.log(x) - _log_beta(b, a)
    ) * _beta_continued_fraction(b, a, 1.0 - x) / b


def t_two_sided_p(t_value: float, df: float) -> float:
    """Probability of a t statistic at least this far from zero, either way."""
    if df <= 0 or not np.isfinite(t_value):
        return float("nan")
    t_value = abs(float(t_value))
    if t_value == 0.0:
        return 1.0
    return regularized_incomplete_beta(df / 2.0, 0.5, df / (df + t_value * t_value))


def t_critical(df: float, confidence: float = 0.95) -> float:
    """The t value with (1 - confidence) probability split between both tails.

    Found by bisection on the two sided tail probability. Bisection is slower
    than a closed form approximation and it cannot go wrong, which is the right
    trade for a function called a few hundred times.
    """
    if df <= 0:
        return float("nan")
    target = 1.0 - confidence

    low, high = 0.0, 1000.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if t_two_sided_p(middle, df) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def normal_critical(confidence: float = 0.95) -> float:
    """The z value with (1 - confidence) probability split between both tails."""
    # A normal is a t with infinite degrees of freedom. Using a very large df
    # keeps one implementation instead of two.
    return t_critical(1.0e7, confidence)


# ---------------------------------------------------------------------------
# Building a design matrix
# ---------------------------------------------------------------------------

def design_matrix(frame: pd.DataFrame, terms: list[str]) -> tuple[np.ndarray, list[str]]:
    """Turn a list of model terms into a numeric matrix with an intercept.

    A term is either a column name, used as a number, or "C(column)", which is
    expanded into indicator columns with the first category dropped. Dropping
    one category is what makes the intercept interpretable: every coefficient
    is then a difference from that first category rather than a free floating
    level.
    """
    columns: list[np.ndarray] = [np.ones(len(frame))]
    names: list[str] = ["Intercept"]

    for term in terms:
        if term.startswith("C(") and term.endswith(")"):
            column = term[2:-1]
            categories = frame[column].astype(str)
            # Sorted so the reference category does not depend on row order.
            levels = sorted(categories.unique())
            for level in levels[1:]:
                columns.append((categories == level).to_numpy(dtype=float))
                names.append(f"{column}[{level}]")
        else:
            columns.append(pd.to_numeric(frame[term], errors="coerce").to_numpy(dtype=float))
            names.append(term)

    return np.column_stack(columns), names


# ---------------------------------------------------------------------------
# Least squares
# ---------------------------------------------------------------------------

@dataclass
class RegressionResult:
    """Everything the study reports about one fitted model."""

    names: list[str]
    params: np.ndarray
    std_errors: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    p_values: np.ndarray
    n: int
    rank: int
    df: float
    r_squared: float
    adj_r_squared: float

    def for_term(self, name: str) -> dict | None:
        """Pull out one coefficient by name."""
        if name not in self.names:
            return None
        index = self.names.index(name)
        return {
            "coefficient": float(self.params[index]),
            "std_error": float(self.std_errors[index]),
            "ci_low": float(self.ci_low[index]),
            "ci_high": float(self.ci_high[index]),
            "p_value": float(self.p_values[index]),
            "n": self.n,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
        }


def fit_ols(
    y: np.ndarray,
    X: np.ndarray,
    names: list[str],
    clusters: np.ndarray | None = None,
    confidence: float = 0.95,
) -> RegressionResult:
    """Least squares with robust standard errors.

    Without clusters this uses HC3, the heteroskedasticity robust estimator
    that behaves best in small samples. The spread of a school outcome around
    the fitted line is wider for a small school than a large one, which is
    exactly the situation robust errors exist for.

    With clusters it uses the cluster robust estimator, which additionally
    allows observations inside a cluster to be correlated with each other.
    New York City puts several schools in one building, and schools in a
    building share a facility, a neighbourhood, and often the same applicants,
    so treating them as independent would make the intervals too narrow.

    The small sample corrections and the degrees of freedom follow the same
    conventions statsmodels uses, so the numbers match what the earlier version
    of this study reported.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    # pinv rather than an explicit inverse, so a rank deficient design (two
    # controls that carry the same information) degrades instead of crashing.
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    params = XtX_inv @ (X.T @ y)
    rank = int(np.linalg.matrix_rank(X))

    residuals = y - X @ params

    if clusters is None:
        # HC3. Each squared residual is inflated by its leverage, which is what
        # separates HC3 from the simpler HC0.
        leverage = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
        leverage = np.clip(leverage, 0.0, 1.0 - 1.0e-10)
        weights = (residuals / (1.0 - leverage)) ** 2
        meat = (X * weights[:, None]).T @ X
        covariance = XtX_inv @ meat @ XtX_inv
        df = float(n - rank)
    else:
        codes = pd.Series(clusters).astype(str).to_numpy()
        unique = np.unique(codes)
        meat = np.zeros_like(XtX)
        for group in unique:
            mask = codes == group
            scores = X[mask].T @ residuals[mask]
            meat += np.outer(scores, scores)

        groups = len(unique)
        correction = (groups / (groups - 1.0)) * ((n - 1.0) / (n - rank)) if groups > 1 else 1.0
        covariance = XtX_inv @ meat @ XtX_inv * correction
        # With clustering the effective sample size is the number of clusters,
        # not the number of rows, so the degrees of freedom follow the clusters.
        df = float(groups - 1)

    variances = np.clip(np.diag(covariance), 0.0, None)
    std_errors = np.sqrt(variances)

    critical = t_critical(df, confidence)
    ci_low = params - critical * std_errors
    ci_high = params + critical * std_errors

    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistics = np.divide(params, std_errors)
    p_values = np.array(
        [t_two_sided_p(value, df) if np.isfinite(value) else np.nan for value in t_statistics]
    )

    total_sum_squares = float(((y - y.mean()) ** 2).sum())
    residual_sum_squares = float((residuals**2).sum())
    r_squared = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else np.nan
    if n - rank > 0 and total_sum_squares > 0:
        adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1.0) / (n - rank)
    else:
        adj_r_squared = np.nan

    return RegressionResult(
        names=names,
        params=params,
        std_errors=std_errors,
        ci_low=ci_low,
        ci_high=ci_high,
        p_values=p_values,
        n=n,
        rank=rank,
        df=df,
        r_squared=float(r_squared),
        adj_r_squared=float(adj_r_squared),
    )


# ---------------------------------------------------------------------------
# Correlation and the two sample test
# ---------------------------------------------------------------------------

def pearson_correlation(x, y, confidence: float = 0.95) -> dict:
    """Pearson correlation with a confidence interval and a p value.

    The interval is built with the Fisher z transformation. A correlation is
    bounded at minus one and one, so its sampling distribution is skewed near
    the ends. Fisher z maps it to a scale where the distribution is close to
    normal, the interval is built there, and the ends are mapped back. Doing it
    the naive way produces intervals that run past one.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 4:
        return {"r": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "n": n}

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = math.sqrt(float((x_centered**2).sum()) * float((y_centered**2).sum()))
    if denominator == 0:
        return {"r": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "n": n}

    r = float((x_centered * y_centered).sum() / denominator)
    r = max(-1.0, min(1.0, r))

    if abs(r) >= 1.0:
        p_value = 0.0
    else:
        t_statistic = r * math.sqrt((n - 2) / (1.0 - r * r))
        p_value = t_two_sided_p(t_statistic, n - 2)

    z = math.atanh(r) if abs(r) < 1.0 else math.copysign(float("inf"), r)
    standard_error = 1.0 / math.sqrt(n - 3)
    critical = normal_critical(confidence)

    return {
        "r": r,
        "ci_low": math.tanh(z - critical * standard_error),
        "ci_high": math.tanh(z + critical * standard_error),
        "p_value": float(p_value),
        "n": n,
    }


def welch_test(a, b, confidence: float = 0.95) -> dict:
    """Welch two sample test of the difference in means, a minus b.

    Welch rather than the equal variance version, because the two groups being
    compared here are often very different sizes and there is no reason to
    assume they have the same spread.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {"difference": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}

    var_a = float(a.var(ddof=1)) / len(a)
    var_b = float(b.var(ddof=1)) / len(b)
    standard_error = math.sqrt(var_a + var_b)
    difference = float(a.mean() - b.mean())

    if standard_error == 0:
        return {"difference": difference, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}

    # The Welch-Satterthwaite degrees of freedom.
    df = (var_a + var_b) ** 2 / (
        var_a**2 / (len(a) - 1) + var_b**2 / (len(b) - 1)
    )
    critical = t_critical(df, confidence)

    return {
        "difference": difference,
        "ci_low": difference - critical * standard_error,
        "ci_high": difference + critical * standard_error,
        "p_value": float(t_two_sided_p(difference / standard_error, df)),
        "df": float(df),
    }
