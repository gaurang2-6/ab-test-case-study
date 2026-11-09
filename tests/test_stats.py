import numpy as np
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from abtest import stats as s


def test_z_test_matches_statsmodels():
    res = s.two_proportion_test(1200, 10_000, 1290, 10_000)
    z, p = proportions_ztest([1290, 1200], [10_000, 10_000])
    assert np.isclose(res.z, z) and np.isclose(res.p_value, p)
    assert res.ci_low < res.diff < res.ci_high


def test_sample_size_and_mde_are_inverses():
    n = s.sample_size_per_group(0.12, 0.005)
    assert 60_000 < n < 80_000                       # textbook value for 12% → 12.5% is ~67k
    assert np.isclose(s.minimum_detectable_effect(n, 0.12), 0.005, atol=2e-5)


def test_srm_flags_a_skewed_split_only():
    assert s.srm_test([50_000, 50_100]) > 0.05
    assert s.srm_test([50_000, 51_500]) < 0.001
    assert s.srm_test([20_000, 80_000], [0.2, 0.8]) > 0.05


def test_holm_matches_statsmodels():
    p = [0.01, 0.04, 0.03, 0.2]
    assert np.allclose(s.holm(p), multipletests(p, method="holm")[1])


def test_ci_covers_true_difference_about_95_percent():
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(1000):
        xa, xb = rng.binomial(5000, 0.10), rng.binomial(5000, 0.12)
        r = s.two_proportion_test(xa, 5000, xb, 5000)
        hits += r.ci_low <= 0.02 <= r.ci_high
    assert 0.93 < hits / 1000 < 0.97


def test_bayes_probability_is_symmetric():
    p_ab, _ = s.prob_b_beats_a(100, 1000, 120, 1000)
    p_ba, _ = s.prob_b_beats_a(120, 1000, 100, 1000)
    assert np.isclose(p_ab + p_ba, 1, atol=0.01) and p_ab > 0.9
