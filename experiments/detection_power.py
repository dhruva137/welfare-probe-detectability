"""Sample-size and power calculation for the planned detection-rate comparison.

The planned study compares detection rate (fraction of items a judge model
classifies as "evaluation") between 40 welfare items and 40 comparison items,
each judged under 3 prompt variants with 3 samples per variant
(src/judge.py::JUDGE_PROMPT_VARIANTS, majority_vote). This script asks the
question that should be answered *before* that study runs, not after: at that
size, what difference in detection rate can actually be detected?

Unit of analysis. The 3 samples within a (item, variant) cell are collapsed by
majority vote before analysis (src.judge.majority_vote), so they are not 3
independent data points -- they are 3 draws that produce one point estimate
per item per variant. The 3 prompt variants are the closest thing to
independent replication the design has. This script therefore treats
`n_items * n_variants` as the effective number of quasi-independent Bernoulli
trials per group. That is an approximation: items are still shared across
variants, so true independence is lower than this treats it as. This is
flagged, not hidden, and is exactly the kind of design-effect issue a mentor
with psychometrics experience would be expected to press on -- the honest
fix is a clustered/mixed-effects analysis at run time, not a bigger effective
n at planning time.

Power is computed via the standard normal approximation to the two-proportion
z-test (e.g. Fleiss, Levin & Paik, *Statistical Methods for Rates and
Proportions*), implemented directly rather than via statsmodels, which is
outside this repo's Class A dependency budget (numpy, pandas, scikit-learn,
matplotlib, scipy; see 00_BUILD_RULES.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def two_proportion_power(
    p1: float, p2: float, n_per_group: int, alpha: float = 0.05,
) -> float:
    """Power of a two-sided two-proportion z-test at a given per-group n.

    Normal approximation: reject H0: p1=p2 if the observed difference exceeds
    the critical value under the pooled null variance. Power is the
    probability that happens given the true p1, p2.
    """
    if n_per_group <= 0:
        return 0.0
    p_bar = (p1 + p2) / 2.0
    se_null = np.sqrt(2 * p_bar * (1 - p_bar) / n_per_group)
    se_alt = np.sqrt(p1 * (1 - p1) / n_per_group + p2 * (1 - p2) / n_per_group)
    if se_null == 0 or se_alt == 0:
        return 1.0 if p1 != p2 else float(alpha)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z = (abs(p1 - p2) - z_alpha * se_null) / se_alt
    return float(stats.norm.cdf(z))


def minimum_detectable_effect(
    p2: float, n_per_group: int, power_target: float = 0.80, alpha: float = 0.05,
    p1_grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Smallest |p1 - p2| reaching power_target at n_per_group, searched over p1 in (p2, 1).

    Returns (p1_at_target, mde) where mde = p1_at_target - p2. Grid search
    rather than a closed form because the normal-approximation power formula
    is monotonic but not trivially invertible once both p1 and p2 (not just
    their difference) matter, as they do here through se_alt.
    """
    if p1_grid is None:
        p1_grid = np.linspace(p2 + 0.001, 0.999, 400)
    for p1 in p1_grid:
        if two_proportion_power(p1, p2, n_per_group, alpha) >= power_target:
            return float(p1), float(p1 - p2)
    return float("nan"), float("nan")


def power_curve_by_item_count(
    item_counts: np.ndarray, n_variants: int, p2: float, effect_sizes: list[float], alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for n_items in item_counts:
        n_per_group = int(n_items) * n_variants
        for delta in effect_sizes:
            p1 = min(p2 + delta, 0.999)
            power = two_proportion_power(p1, p2, n_per_group, alpha)
            rows.append({
                "n_items": int(n_items), "n_per_group_effective": n_per_group,
                "delta": delta, "p1": p1, "p2": p2, "power": power,
            })
    return pd.DataFrame(rows)


def make_figure(curve_table: pd.DataFrame, planned_n_items: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for delta, grp in curve_table.groupby("delta"):
        ax.plot(grp["n_items"], grp["power"], marker="o", markersize=3,
                label=f"delta detection rate = {delta:.2f}")
    ax.axhline(0.80, color="gray", linestyle="--", linewidth=1, label="80% power")
    ax.axvline(planned_n_items, color="black", linestyle=":", linewidth=1,
               label=f"planned n_items = {planned_n_items}")
    ax.set_xlabel("items per group (welfare items = comparison items = n_items)")
    ax.set_ylabel("power (two-sided, alpha=0.05)")
    ax.set_title("Power vs. item count, at 3 prompt variants per item")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-items", type=int, default=40, help="planned items per group")
    parser.add_argument("--n-variants", type=int, default=3, help="prompt variants per item")
    parser.add_argument("--n-samples", type=int, default=3, help="samples per variant (collapsed by majority vote)")
    parser.add_argument("--baseline-rate", type=float, default=0.15,
                         help="assumed comparison-item detection rate p2")
    parser.add_argument("--power-target", type=float, default=0.80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-item-count", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    parser.add_argument("--dry-run", action="store_true",
                         help="exercise the full code path on a tiny grid, <5s")
    args = parser.parse_args()

    if args.dry_run:
        args.max_item_count = 20
        item_counts = np.array([5, 10, 20])
    else:
        item_counts = np.arange(5, args.max_item_count + 1, 5)

    n_per_group_planned = args.n_items * args.n_variants

    print("=" * 78)
    print("DETECTION POWER: sample-size planning for the welfare vs. comparison study")
    print("=" * 78)
    print(f"planned items per group      : {args.n_items}")
    print(f"prompt variants per item     : {args.n_variants} (majority vote over {args.n_samples} samples each)")
    print(f"effective quasi-independent n per group : {n_per_group_planned}")
    print(f"  (see module docstring: this treats item x variant cells as independent,")
    print(f"   which is an optimistic approximation -- items repeat across variants)")
    print(f"assumed baseline detection rate (comparison items), p2 : {args.baseline_rate:.0%}")
    print(f"alpha (two-sided) : {args.alpha}, power target : {args.power_target:.0%}")
    print()

    p1_target, mde = minimum_detectable_effect(
        args.baseline_rate, n_per_group_planned, args.power_target, args.alpha)
    print(f"Minimum detectable effect at planned size:")
    print(f"  welfare-item detection rate must reach >= {p1_target:.3f} "
          f"(i.e. p2 + {mde:.3f}) for {args.power_target:.0%} power to detect it.")
    print()

    # Table: MDE at a range of item counts, holding the design (variants, alpha, power) fixed.
    print("-" * 78)
    print("MINIMUM DETECTABLE EFFECT vs. ITEM COUNT")
    print("-" * 78)
    mde_rows = []
    for n_items in item_counts:
        n_pg = int(n_items) * args.n_variants
        p1_t, mde_t = minimum_detectable_effect(args.baseline_rate, n_pg, args.power_target, args.alpha)
        mde_rows.append({"n_items": int(n_items), "n_per_group_effective": n_pg,
                          "min_detectable_p1": p1_t, "min_detectable_effect": mde_t})
    mde_table = pd.DataFrame(mde_rows)
    print_mde = mde_table.copy()
    print_mde["min_detectable_p1"] = print_mde["min_detectable_p1"].map(lambda v: f"{v:.3f}")
    print_mde["min_detectable_effect"] = print_mde["min_detectable_effect"].map(lambda v: f"{v:.3f}")
    print(print_mde.to_string(index=False))
    print()

    print("-" * 78)
    print("POWER CURVE vs. ITEM COUNT, for representative effect sizes")
    print("-" * 78)
    effect_sizes = [0.15, 0.25, 0.35]
    curve_table = power_curve_by_item_count(item_counts, args.n_variants, args.baseline_rate, effect_sizes, args.alpha)
    pivot = curve_table.pivot(index="n_items", columns="delta", values="power")
    pivot.columns = [f"power(delta={c:.2f})" for c in pivot.columns]
    print(pivot.round(3).to_string())

    out_path = args.output_dir / "power_curve.png"
    full_curve = power_curve_by_item_count(
        np.arange(5, args.max_item_count + 1, 5), args.n_variants, args.baseline_rate, effect_sizes, args.alpha)
    make_figure(full_curve, args.n_items, out_path)
    print()
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
