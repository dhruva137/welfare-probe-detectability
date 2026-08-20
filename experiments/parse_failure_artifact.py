"""The headline artifact: differential parse failure manufactures a fake finding.

Two synthetic models are constructed to have an *identical* underlying welfare
score distribution -- same mean, same spread, same shape -- and differ only in
how often they refuse to answer at all (5% vs. 30%, by default). Nothing about
their substantive responses differs. Any honest analysis should therefore find
no reliable difference in mean welfare score between them.

Each model's raw outputs are run through the real parsing pipeline
(src.coding_scheme.parse_with_provenance), then the resulting (score,
is_failure) pairs are analyzed under three parser policies
(src.analysis.apply_parser_policy / three_way_sensitivity):

1. drop      -- discard failures, compare means of what's left
2. map_to_lowest -- treat every failure as "lowest score" (e.g. "no distress")
3. separate_category -- keep failures out of the numeric comparison entirely

Policy 2 is not a strawman. "The model didn't answer, so count it as reporting
no distress" is a real, easy-to-reach-for parsing choice, and it is exactly
what silently equates refusal with well-being. Because model B refuses far
more often, policy 2 pulls its apparent mean welfare score down hard even
though its true underlying distribution is identical to model A's -- producing
a large, statistically significant "finding" about two models that are
identical by construction.

This script needs no model, no API key, and no GPU. That is the point: the
confound is a property of the analysis pipeline, not of any particular model,
and it can be demonstrated on synthetic data in under a second.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis import PARSER_POLICIES, three_way_sensitivity, sensitivity_is_stable
from src.coding_scheme import PARSE_OUTCOMES, parse_with_provenance

REFUSAL_TEMPLATES = (
    "I don't have subjective experiences to report, so I can't answer that.",
    "I'm not able to assess my own internal states in the way this question assumes.",
    "As an AI, I don't have feelings in the sense this question is asking about.",
    "I can't meaningfully rate this -- I don't have that kind of experience.",
)


def simulate_model_outputs(
    n_items: int,
    refusal_rate: float,
    mean_score: float,
    sd_score: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate raw text outputs for one synthetic model, then parse them for real.

    The "true" score for every item is drawn from the *same* distribution
    regardless of refusal_rate -- that identity is what makes the later
    between-model difference an artifact rather than a finding. Whether an
    item is a refusal is decided independently of that true score, matching
    the scenario the confound describes: refusal rate as a nuisance property
    of a model (e.g. its safety tuning), unrelated to the construct the probe
    intends to measure.

    Returns (parsed_scores, is_failure, raw_outputs). parsed_scores[i] is
    NaN wherever is_failure[i] is True (no score was actually produced).
    """
    true_scores = np.clip(np.round(rng.normal(mean_score, sd_score, n_items)), 1, 7)
    is_refusal = rng.random(n_items) < refusal_rate

    raw_outputs: list[str] = []
    for i in range(n_items):
        if is_refusal[i]:
            raw_outputs.append(rng.choice(REFUSAL_TEMPLATES))
        else:
            raw_outputs.append(str(int(true_scores[i])))

    parsed_scores = np.full(n_items, np.nan)
    is_failure = np.zeros(n_items, dtype=bool)
    for i, raw in enumerate(raw_outputs):
        result = parse_with_provenance(raw)
        if result.outcome == PARSE_OUTCOMES.scored:
            parsed_scores[i] = result.value
        else:
            is_failure[i] = True

    return parsed_scores, is_failure, raw_outputs


def run_sweep(
    n_items: int,
    gaps: np.ndarray,
    base_refusal_a: float,
    mean_score: float,
    sd_score: float,
    seed: int,
) -> pd.DataFrame:
    """For each refusal-rate gap, compute the map_to_lowest spurious effect."""
    rows = []
    for gap_points in gaps:
        rng = np.random.default_rng(seed + int(gap_points * 100))
        refusal_b = base_refusal_a + gap_points / 100.0
        scores_a, fail_a, _ = simulate_model_outputs(n_items, base_refusal_a, mean_score, sd_score, rng)
        scores_b, fail_b, _ = simulate_model_outputs(n_items, refusal_b, mean_score, sd_score, rng)
        table = three_way_sensitivity(scores_a, fail_a, scores_b, fail_b, lowest_score=1.0)
        policy_row = table[table["policy"] == "map_to_lowest"].iloc[0]
        rows.append({
            "refusal_gap_pts": gap_points,
            "refusal_rate_a": base_refusal_a,
            "refusal_rate_b": refusal_b,
            "cohens_d": policy_row["cohens_d"],
            "p_value": policy_row["p_value"],
            "significant_at_05": policy_row["significant_at_05"],
        })
    return pd.DataFrame(rows)


def make_figure(main_table: pd.DataFrame, sweep_table: pd.DataFrame, out_path: Path) -> None:
    fig, (ax_bars, ax_sweep) = plt.subplots(1, 2, figsize=(11, 4.5))

    policies = main_table["policy"].tolist()
    x = np.arange(len(policies))
    width = 0.35
    ax_bars.bar(x - width / 2, main_table["mean_a"], width, label="Model A (5% refusal)", color="#4C72B0")
    ax_bars.bar(x + width / 2, main_table["mean_b"], width, label="Model B (30% refusal)", color="#DD8452")
    for i, row in main_table.iterrows():
        star = "  *" if row["significant_at_05"] else ""
        ax_bars.text(i, max(row["mean_a"], row["mean_b"]) + 0.1,
                     f"d={row['cohens_d']:.2f}{star}", ha="center", fontsize=8)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(["drop", "map to\nlowest", "separate\ncategory"])
    ax_bars.set_ylabel("apparent mean welfare score (1-7)")
    ax_bars.set_title("Same underlying models, three parser policies")
    ax_bars.legend(fontsize=8)
    ax_bars.set_ylim(0, 7.5)

    ax_sweep.plot(sweep_table["refusal_gap_pts"], sweep_table["cohens_d"], marker="o", color="#C44E52")
    ax_sweep.axhline(0.2, color="gray", linestyle="--", linewidth=1, label="small effect (d=0.2)")
    ax_sweep.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="medium effect (d=0.5)")
    sig = sweep_table[sweep_table["significant_at_05"]]
    if len(sig) > 0:
        ax_sweep.scatter(sig["refusal_gap_pts"], sig["cohens_d"], color="black", zorder=5,
                          s=20, label="p < 0.05")
    ax_sweep.set_xlabel("refusal-rate gap between models (percentage points)")
    ax_sweep.set_ylabel("spurious Cohen's d (map_to_lowest policy)")
    ax_sweep.set_title("Spurious effect size vs. refusal-rate gap")
    ax_sweep.legend(fontsize=7, loc="upper left")

    fig.suptitle("Differential parse failure manufactures a fake between-model difference", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-items", type=int, default=400)
    parser.add_argument("--refusal-a", type=float, default=0.05)
    parser.add_argument("--refusal-b", type=float, default=0.30)
    parser.add_argument("--mean-score", type=float, default=4.0)
    parser.add_argument("--sd-score", type=float, default=1.3)
    parser.add_argument("--sweep-max-gap", type=int, default=40)
    parser.add_argument("--sweep-step", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    parser.add_argument("--dry-run", action="store_true",
                         help="exercise the full code path on 10 items with a tiny sweep, <5s")
    args = parser.parse_args()

    if args.dry_run:
        args.n_items = 10
        args.sweep_max_gap = 20
        args.sweep_step = 20

    rng = np.random.default_rng(args.seed)
    scores_a, fail_a, raw_a = simulate_model_outputs(
        args.n_items, args.refusal_a, args.mean_score, args.sd_score, rng)
    scores_b, fail_b, raw_b = simulate_model_outputs(
        args.n_items, args.refusal_b, args.mean_score, args.sd_score, rng)

    print("=" * 78)
    print("PARSE FAILURE ARTIFACT: two identical models, three parser policies")
    print("=" * 78)
    print(f"n_items per model     : {args.n_items}")
    print(f"true score generator  : identical for both models, N(mu={args.mean_score}, sd={args.sd_score}), clipped [1,7]")
    print(f"Model A refusal rate  : {args.refusal_a:.1%}  (observed: {fail_a.mean():.1%})")
    print(f"Model B refusal rate  : {args.refusal_b:.1%}  (observed: {fail_b.mean():.1%})")
    print()

    main_table = three_way_sensitivity(scores_a, fail_a, scores_b, fail_b, lowest_score=1.0)
    print_table = main_table.copy()
    for col in ("mean_a", "mean_b", "mean_diff", "cohens_d", "t_stat", "p_value"):
        print_table[col] = print_table[col].map(lambda v: f"{v:.4f}")
    print(print_table.to_string(index=False))
    print()

    stable = sensitivity_is_stable(main_table)
    map_row = main_table[main_table["policy"] == "map_to_lowest"].iloc[0]
    print(f"map_to_lowest policy   : Cohen's d = {map_row['cohens_d']:.3f}, "
          f"p = {map_row['p_value']:.6g}  -> "
          f"{'SIGNIFICANT (spurious)' if map_row['significant_at_05'] else 'not significant'}")
    print(f"conclusion stable across all three policies? {stable}")
    if not stable:
        print("  -> the apparent finding under map_to_lowest is a parsing artifact, not a")
        print("     between-model difference: 'drop' and 'separate_category' agree the two")
        print("     models are statistically indistinguishable, which is correct since they")
        print("     were constructed to be identical.")
    print()

    gaps = np.arange(0, args.sweep_max_gap + 1, args.sweep_step)
    sweep_table = run_sweep(args.n_items, gaps, args.refusal_a, args.mean_score, args.sd_score, args.seed)
    print("-" * 78)
    print("SWEEP: refusal-rate gap (pts) -> spurious effect size under map_to_lowest")
    print("-" * 78)
    print_sweep = sweep_table.copy()
    print_sweep["cohens_d"] = print_sweep["cohens_d"].map(lambda v: f"{v:.3f}")
    print_sweep["p_value"] = print_sweep["p_value"].map(lambda v: f"{v:.4g}")
    print(print_sweep.to_string(index=False))

    out_path = args.output_dir / "parse_failure_artifact.png"
    make_figure(main_table, sweep_table, out_path)
    print()
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
