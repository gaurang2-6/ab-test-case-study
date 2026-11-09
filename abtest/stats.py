"""Statistics for two-group conversion experiments, written out so every step is visible."""
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ProportionTest:
    n_a: int
    n_b: int
    rate_a: float
    rate_b: float
    diff: float          # rate_b - rate_a (absolute, as a proportion)
    rel_lift: float      # diff / rate_a
    z: float
    p_value: float
    ci_low: float        # CI for diff
    ci_high: float


def sample_size_per_group(p_base: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users per group to detect an absolute change of `mde_abs` from `p_base` (two-sided z-test)."""
    p_alt = p_base + mde_abs
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p_base + p_alt) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p_base * (1 - p_base) + p_alt * (1 - p_alt))) ** 2
    return int(np.ceil(num / mde_abs ** 2))


def minimum_detectable_effect(n_per_group: int, p_base: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute lift detectable with `n_per_group` users per group (inverse of sample_size_per_group)."""
    lo, hi = 1e-6, 1 - p_base - 1e-6
    for _ in range(100):  # bisection: sample size falls monotonically as the effect grows
        mid = (lo + hi) / 2
        if sample_size_per_group(p_base, mid, alpha, power) > n_per_group:
            lo = mid
        else:
            hi = mid
    return hi


def srm_test(counts, expected_shares=None) -> float:
    """Sample-ratio mismatch: chi-square p-value that the observed split matches the planned one."""
    counts = np.asarray(counts, dtype=float)
    shares = np.full(len(counts), 1 / len(counts)) if expected_shares is None else np.asarray(expected_shares)
    return float(stats.chisquare(counts, counts.sum() * shares).pvalue)


def two_proportion_test(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05) -> ProportionTest:
    """Two-sided z-test (pooled SE under H0) plus a Wald CI for the difference (unpooled SE)."""
    p_a, p_b = x_a / n_a, x_b / n_b
    diff = p_b - p_a
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pool
    p_value = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    half = stats.norm.ppf(1 - alpha / 2) * se
    return ProportionTest(n_a, n_b, p_a, p_b, diff, diff / p_a, float(z), float(p_value), diff - half, diff + half)


def prob_b_beats_a(x_a: int, n_a: int, x_b: int, n_b: int, draws: int = 200_000, seed: int = 0):
    """Bayesian view with flat Beta(1, 1) priors: P(rate_b > rate_a) and the expected loss of choosing B."""
    rng = np.random.default_rng(seed)
    a = rng.beta(1 + x_a, 1 + n_a - x_a, draws)
    b = rng.beta(1 + x_b, 1 + n_b - x_b, draws)
    return float((b > a).mean()), float(np.maximum(a - b, 0).mean())


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of mean(b) - mean(a)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    # Resampling 0/1 outcomes is the same as drawing binomial counts, which is much faster.
    if set(np.unique(a)) <= {0, 1} and set(np.unique(b)) <= {0, 1}:
        return rng.binomial(len(b), b.mean(), n_boot) / len(b) - rng.binomial(len(a), a.mean(), n_boot) / len(a)
    return np.array([rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(n_boot)])


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted
