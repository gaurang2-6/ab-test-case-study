"""Experiment 1: new vs. old landing page (conversion)."""
import numpy as np
import pandas as pd

from abtest import data, stats as s

BASELINE = 0.12          # pre-test conversion rate the team planned around
MDE_PLANNED = 0.005      # smallest lift worth shipping: +0.5 pt (≈ +4% relative)


def clean(raw: pd.DataFrame):
    """Drop rows where the group and the page disagree, then duplicate users. Returns (df, log)."""
    log = [("Raw rows", len(raw))]
    aligned = raw[(raw.group == "treatment") == (raw.landing_page == "new_page")]
    log.append(("Group and page agree (control→old, treatment→new)", len(aligned)))
    df = aligned.sort_values("timestamp").drop_duplicates("user_id", keep="first")
    log.append(("One row per user (first exposure kept)", len(df)))
    return df.reset_index(drop=True), pd.DataFrame(log, columns=["step", "rows"])


def analyse() -> dict:
    df, log = clean(data.landing_page_raw())
    g = df.groupby("group").converted.agg(["sum", "count"])
    xa, na = int(g.loc["control", "sum"]), int(g.loc["control", "count"])
    xb, nb = int(g.loc["treatment", "sum"]), int(g.loc["treatment", "count"])

    need = s.sample_size_per_group(BASELINE, MDE_PLANNED)
    test = s.two_proportion_test(xa, na, xb, nb)
    p_better, loss = s.prob_b_beats_a(xa, na, xb, nb)
    boot = s.bootstrap_diff(df.loc[df.group == "control", "converted"], df.loc[df.group == "treatment", "converted"])

    # Day by day: does the new page start strong and fade (novelty), or the reverse?
    df["day"] = df.timestamp.dt.normalize()
    daily = (df.groupby(["day", "group"]).converted.agg(["sum", "count"]).unstack("group"))
    daily = pd.DataFrame({
        "users": daily["count"].sum(axis=1),
        "control_rate": daily["sum"]["control"] / daily["count"]["control"],
        "treatment_rate": daily["sum"]["treatment"] / daily["count"]["treatment"],
    })
    daily["diff"] = daily.treatment_rate - daily.control_rate
    cum = df.groupby(["day", "group"]).converted.agg(["sum", "count"]).unstack("group").cumsum()
    ca, ta = cum["sum"]["control"] / cum["count"]["control"], cum["sum"]["treatment"] / cum["count"]["treatment"]
    se = np.sqrt(ca * (1 - ca) / cum["count"]["control"] + ta * (1 - ta) / cum["count"]["treatment"])
    daily["cum_diff"], daily["cum_low"], daily["cum_high"] = ta - ca, ta - ca - 1.96 * se, ta - ca + 1.96 * se

    # Country segments, Holm-corrected because we are now running three tests instead of one.
    seg = []
    for country, d in df.groupby("country"):
        c, t = d[d.group == "control"].converted, d[d.group == "treatment"].converted
        r = s.two_proportion_test(int(c.sum()), len(c), int(t.sum()), len(t))
        seg.append({"country": country, "users": len(d), "control_rate": r.rate_a, "treatment_rate": r.rate_b,
                    "diff": r.diff, "ci_low": r.ci_low, "ci_high": r.ci_high, "p_value": r.p_value})
    seg = pd.DataFrame(seg).sort_values("users", ascending=False)
    seg["p_holm"] = s.holm(seg.p_value)

    return {
        "cleaning_log": log,
        "dates": (df.timestamp.min(), df.timestamp.max()),
        "srm_p": s.srm_test([na, nb]),
        "required_per_group": need,
        "mde_achieved": s.minimum_detectable_effect(min(na, nb), xa / na),
        "test": test,
        "prob_better": p_better,
        "expected_loss": loss,
        "boot_ci": tuple(np.percentile(boot, [2.5, 97.5])),
        "daily": daily,
        "segments": seg,
    }
