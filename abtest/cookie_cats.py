"""Experiment 2: Cookie Cats, moving the first progress gate from level 30 to level 40 (retention)."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from abtest import data, stats as s


def analyse() -> dict:
    df = data.cookie_cats()
    a, b = df[df.version == "gate_30"], df[df.version == "gate_40"]
    na, nb = len(a), len(b)

    results = {}
    for metric in ["retention_1", "retention_7"]:
        xa, xb = int(a[metric].sum()), int(b[metric].sum())
        test = s.two_proportion_test(xa, na, xb, nb)
        p_b_better, _ = s.prob_b_beats_a(xa, na, xb, nb)
        boot = s.bootstrap_diff(a[metric], b[metric], n_boot=5000)
        results[metric] = {"test": test, "prob_gate40_better": p_b_better, "boot": boot,
                           "boot_ci": tuple(np.percentile(boot, [2.5, 97.5]))}

    # Engagement: heavy-tailed (one player has ~50k rounds), so compare medians and use Mann-Whitney.
    rounds = pd.DataFrame({
        "gate": ["gate_30", "gate_40"],
        "players": [na, nb],
        "zero_rounds_pct": [(a.sum_gamerounds == 0).mean(), (b.sum_gamerounds == 0).mean()],
        "median_rounds": [a.sum_gamerounds.median(), b.sum_gamerounds.median()],
        "p99_rounds": [a.sum_gamerounds.quantile(0.99), b.sum_gamerounds.quantile(0.99)],
        "max_rounds": [a.sum_gamerounds.max(), b.sum_gamerounds.max()],
    })
    mw_p = float(mannwhitneyu(a.sum_gamerounds, b.sum_gamerounds).pvalue)

    return {
        "n": (na, nb),
        "srm_p": s.srm_test([na, nb]),
        "split_pct": na / (na + nb),
        "metrics": results,
        "rounds": rounds,
        "rounds_mw_p": mw_p,
        "mde_r7": s.minimum_detectable_effect(min(na, nb), a.retention_7.mean()),
        "mde_r1": s.minimum_detectable_effect(min(na, nb), a.retention_1.mean()),
    }
