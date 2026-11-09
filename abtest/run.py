"""Run both experiments, write reports/results.md and the charts in reports/figures/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from abtest import cookie_cats, landing_page, stats as s

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
BLUE, ORANGE, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "legend.frameon": False, "lines.linewidth": 2,
})


def pts(x):
    return f"{x * 100:+.2f} pt"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(fig)


def fig_power(lp):
    """How many users the test needed, and what effect the users it got can detect."""
    mdes = np.linspace(0.002, 0.012, 200)
    n = [s.sample_size_per_group(landing_page.BASELINE, m) for m in mdes]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(mdes * 100, np.array(n) / 1000, color=BLUE)
    got = lp["test"].n_a
    ax.axhline(got / 1000, color=INK2, lw=1, ls="--")
    ax.text(1.18, got / 1000 + 6, f"Users per group in the test: {got / 1000:.0f}k", ha="right", color=INK2)
    ax.plot([landing_page.MDE_PLANNED * 100], [lp["required_per_group"] / 1000], "o", ms=8, color=BLUE,
            mec=SURFACE, mew=2)
    ax.annotate(f"Planned: +0.5 pt needs {lp['required_per_group'] / 1000:.1f}k per group",
                (landing_page.MDE_PLANNED * 100, lp["required_per_group"] / 1000), xytext=(12, 4),
                textcoords="offset points", color=INK)
    ax.plot([lp["mde_achieved"] * 100], [got / 1000], "o", ms=8, color=ORANGE, mec=SURFACE, mew=2)
    ax.annotate(f"Achieved MDE: +{lp['mde_achieved'] * 100:.2f} pt", (lp["mde_achieved"] * 100, got / 1000),
                xytext=(10, -16), textcoords="offset points", color=INK)
    ax.set(ylim=(0, 300), xlabel="Minimum detectable effect (absolute, percentage points)",
           ylabel="Users needed per group (thousands)")
    ax.set_title("Power analysis: the landing-page test had enough traffic", pad=12)
    save(fig, "power_curve")


def fig_cumulative(lp):
    d = lp["daily"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.fill_between(d.index, d.cum_low * 100, d.cum_high * 100, color=BLUE, alpha=0.18, lw=0, label="95% CI")
    ax.plot(d.index, d.cum_diff * 100, color=BLUE, label="Cumulative difference (new − old)")
    ax.axhline(0, color=INK2, lw=1)
    ax.axhline(landing_page.MDE_PLANNED * 100, color=ORANGE, lw=1.5, ls="--")
    ax.text(d.index[-1], landing_page.MDE_PLANNED * 100 + 0.04, "Smallest lift worth shipping (+0.5 pt)",
            ha="right", color=INK2)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set(ylabel="Conversion difference (pt)", ylim=(-1.6, 0.8))
    ax.legend(loc="lower right")
    ax.set_title("The new page never got close to the target lift", pad=12)
    save(fig, "landing_cumulative_diff")


def fig_forest(lp):
    rows = [("All users", lp["test"].diff, lp["test"].ci_low, lp["test"].ci_high)]
    rows += [(r.country, r.diff, r.ci_low, r.ci_high) for r in lp["segments"].itertuples()]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, (label, d, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo * 100, hi * 100], [y, y], color=BLUE if i == 0 else INK2, lw=2, solid_capstyle="round")
        ax.plot(d * 100, y, "o", ms=8, color=BLUE if i == 0 else INK2, mec=SURFACE, mew=2)
        ax.text(1.35, y, pts(d), va="center", ha="right", color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.axvline(landing_page.MDE_PLANNED * 100, color=ORANGE, lw=1.5, ls="--")
    ax.text(landing_page.MDE_PLANNED * 100 + 0.04, len(rows) - 0.55, "Target +0.5 pt", color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Conversion difference, new − old (pt, 95% CI)", xlim=(-2, 1.4), ylim=(-0.5, len(rows) - 0.3))
    ax.set_title("No country shows a lift either", pad=12)
    save(fig, "landing_segments")


def fig_cookie(cc):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for metric, color, label in [("retention_1", BLUE, "1-day retention"), ("retention_7", ORANGE, "7-day retention")]:
        boot = cc["metrics"][metric]["boot"] * 100
        bins = np.linspace(-1.8, 0.6, 73)
        ax.hist(boot, bins=bins, color=color, alpha=0.15, histtype="stepfilled")
        ax.hist(boot, bins=bins, color=color, histtype="step", lw=2, label=label)
        lo, hi = cc["metrics"][metric]["boot_ci"]
        right = metric == "retention_1"
        ax.text(0.06 if right else boot.mean() - 0.12, 290,
                f"{label}: {pts(cc['metrics'][metric]['test'].diff)}\n95% CI [{pts(lo)}, {pts(hi)}]",
                ha="left" if right else "right", color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set(xlabel="Retention difference, gate 40 − gate 30 (pt, 5,000 bootstrap resamples)", ylabel="Resamples")
    ax.set_ylim(0, 420)
    ax.legend(loc="upper left")
    ax.set_title("Moving the gate to level 40 lowers 7-day retention", pad=12)
    save(fig, "cookie_cats_bootstrap")


def table(df, fmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt.get(c, str)(v) for c, v in zip(cols, row)) + " |")
    return "\n".join(lines)


def report(lp, cc):
    t = lp["test"]
    pct = lambda v: f"{v * 100:.2f}%"
    out = ["# Results (generated by `python -m abtest.run`)", "", "## Experiment 1: landing page", "",
           table(lp["cleaning_log"], {"rows": lambda v: f"{v:,}"}), "",
           f"- Dates: {lp['dates'][0]:%d %b %Y} to {lp['dates'][1]:%d %b %Y}",
           f"- SRM p-value: {lp['srm_p']:.3f}",
           f"- Required per group for +0.5 pt: {lp['required_per_group']:,}; achieved MDE: {pts(lp['mde_achieved'])}",
           f"- Control {t.n_a:,} users, {pct(t.rate_a)}; treatment {t.n_b:,} users, {pct(t.rate_b)}",
           f"- Difference {pts(t.diff)} (relative {t.rel_lift * 100:+.1f}%), 95% CI [{pts(t.ci_low)}, {pts(t.ci_high)}], "
           f"z = {t.z:.2f}, p = {t.p_value:.3f}",
           f"- Bootstrap 95% CI [{pts(lp['boot_ci'][0])}, {pts(lp['boot_ci'][1])}]",
           f"- P(new page better) = {lp['prob_better']:.1%}; expected loss if shipped = {lp['expected_loss'] * 100:.3f} pt",
           "", table(lp["segments"], {"users": lambda v: f"{v:,}", "control_rate": pct, "treatment_rate": pct,
                                       "diff": pts, "ci_low": pts, "ci_high": pts,
                                       "p_value": lambda v: f"{v:.3f}", "p_holm": lambda v: f"{v:.3f}"}),
           "", "## Experiment 2: Cookie Cats", "",
           f"- Players: gate_30 {cc['n'][0]:,}, gate_40 {cc['n'][1]:,} ({cc['split_pct']:.2%} / {1 - cc['split_pct']:.2%}); "
           f"SRM p-value: {cc['srm_p']:.4f}",
           f"- MDE at this size: 1-day {pts(cc['mde_r1'])}, 7-day {pts(cc['mde_r7'])}", ""]
    for metric, v in cc["metrics"].items():
        r = v["test"]
        out.append(f"- {metric}: {pct(r.rate_a)} → {pct(r.rate_b)}, {pts(r.diff)} (relative {r.rel_lift * 100:+.1f}%), "
                   f"95% CI [{pts(r.ci_low)}, {pts(r.ci_high)}], p = {r.p_value:.4f}, "
                   f"bootstrap CI [{pts(v['boot_ci'][0])}, {pts(v['boot_ci'][1])}], "
                   f"P(gate 40 better) = {v['prob_gate40_better']:.1%}")
    out += ["", table(cc["rounds"], {"players": lambda v: f"{v:,}", "zero_rounds_pct": pct,
                                     "median_rounds": lambda v: f"{v:.0f}", "p99_rounds": lambda v: f"{v:.0f}",
                                     "max_rounds": lambda v: f"{v:,}"}),
            f"", f"- Mann-Whitney p-value for rounds played: {cc['rounds_mw_p']:.3f}", ""]
    (ROOT / "reports" / "results.md").write_text("\n".join(out), encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    lp, cc = landing_page.analyse(), cookie_cats.analyse()
    report(lp, cc)
    fig_power(lp)
    fig_cumulative(lp)
    fig_forest(lp)
    fig_cookie(cc)
    print("Wrote reports/results.md and", len(list(FIG.glob("*.png"))), "figures")


if __name__ == "__main__":
    main()
