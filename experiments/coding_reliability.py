"""Raw percentage agreement overstates reliability when base rates are skewed.

The listing this repo backs explicitly asks for inter-rater reliability
experience, so this script exists to show, not just claim, that the three
statistics in src/reliability.py (percentage agreement, Cohen's kappa,
Krippendorff's alpha) are not interchangeable.

Two synthetic coders rate a binary feature (present/absent) on a set of items.
Their per-item agreement probability is held fixed (e.g. they agree 85% of the
time) while the feature's base rate -- how often it's actually present -- is
swept from rare to common. Raw percentage agreement stays roughly flat across
the sweep, because it was constructed to: the coders were told to agree 85% of
the time, full stop. But kappa and alpha both fall sharply as the base rate
moves toward the extremes, because two coders who mostly say "absent" will
agree most of the time *by chance alone*, and the corrected statistics
subtract that chance agreement back out.

Two coders getting the same six-feature coding scheme in this repo's actual
item bank (src/coding_scheme.py::FEATURES) will very likely hit skewed base
rates on some features (e.g. repeated_battery_structure is probably rare) --
this script is why reporting kappa/alpha per feature, not one pooled number,
is part of the plan rather than an afterthought.

ADJUDICATION PROTOCOL (worked, fixed before any coding starts):

1. Two coders independently code every item on all six FEATURES. Neither
   sees the other's codes until both are complete.
2. Per-feature percentage agreement, kappa, and alpha are computed BEFORE any
   disagreement is resolved. If kappa on a feature falls below 0.60
   ("moderate", Landis & Koch 1977), that feature's definition is treated as
   broken and rewritten -- adjudicating individual items papers over a bad
   definition instead of fixing it.
3. For remaining disagreements, a third, pre-designated adjudicator who did
   not help build the item bank reviews only the disputed feature on the
   disputed item and casts the deciding vote.
4. The adjudicator's decision is final and logged with a one-line reason. No
   retrospective overturning after seeing downstream results.
5. The adjudicated dataset, both original coder datasets, and all reliability
   statistics are all retained; only the adjudicated dataset feeds
   src/analysis.py's regression.

(The same protocol, with the same wording, lives as ADJUDICATION_PROTOCOL in
src/reliability.py so code and documentation cannot drift apart.)
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
from src.reliability import ADJUDICATION_PROTOCOL, cohens_kappa, krippendorff_alpha, percentage_agreement


def simulate_coders(
    n_items: int, base_rate: float, agreement_prob: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Two binary coders with a controllable pairwise agreement probability.

    Coder A's ratings are drawn straight from the base rate. Coder B copies
    coder A's rating with probability `agreement_prob` and flips it otherwise.
    This directly fixes E[percentage_agreement] = agreement_prob regardless of
    base_rate -- the point of the experiment is to show that fixing raw
    agreement this way does NOT fix kappa or alpha, which depend on base_rate
    through the chance-agreement term.
    """
    coder_a = (rng.random(n_items) < base_rate).astype(int)
    flip = rng.random(n_items) >= agreement_prob
    coder_b = coder_a.copy()
    coder_b[flip] = 1 - coder_b[flip]
    return coder_a, coder_b


def sweep_base_rate(
    base_rates: np.ndarray, n_items: int, agreement_prob: float, seed: int,
) -> pd.DataFrame:
    rows = []
    for i, br in enumerate(base_rates):
        rng = np.random.default_rng(seed + i)
        coder_a, coder_b = simulate_coders(n_items, br, agreement_prob, rng)
        pa = percentage_agreement(coder_a.tolist(), coder_b.tolist())
        kappa = cohens_kappa(coder_a.tolist(), coder_b.tolist())
        alpha = krippendorff_alpha([[a, b] for a, b in zip(coder_a.tolist(), coder_b.tolist())])
        rows.append({
            "base_rate": br,
            "n_positive_a": int(coder_a.sum()),
            "percentage_agreement": pa,
            "cohens_kappa": kappa,
            "krippendorff_alpha": alpha,
            "agreement_minus_kappa": pa - kappa,
        })
    return pd.DataFrame(rows)


def make_figure(sweep_table: pd.DataFrame, agreement_prob: float, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(sweep_table["base_rate"], sweep_table["percentage_agreement"],
            marker="o", label="raw percentage agreement", color="#4C72B0")
    ax.plot(sweep_table["base_rate"], sweep_table["cohens_kappa"],
            marker="s", label="Cohen's kappa", color="#C44E52")
    ax.plot(sweep_table["base_rate"], sweep_table["krippendorff_alpha"],
            marker="^", label="Krippendorff's alpha", color="#55A868", linestyle="--")
    ax.axhline(agreement_prob, color="gray", linestyle=":", linewidth=1,
               label=f"coder agreement setting ({agreement_prob:.0%})")
    ax.set_xlabel("feature base rate (fraction of items where feature is present)")
    ax.set_ylabel("statistic value")
    ax.set_title("Raw agreement is flat; chance-corrected statistics are not")
    ax.legend(fontsize=7)
    ax.set_ylim(-0.05, 1.05)

    ax2 = axes[1]
    ax2.plot(sweep_table["base_rate"], sweep_table["agreement_minus_kappa"],
              marker="o", color="#8172B2")
    ax2.fill_between(sweep_table["base_rate"], 0, sweep_table["agreement_minus_kappa"],
                       alpha=0.2, color="#8172B2")
    ax2.set_xlabel("feature base rate")
    ax2.set_ylabel("percentage agreement - Cohen's kappa")
    ax2.set_title("The chance-correction gap widens at skewed base rates")

    fig.suptitle(f"Two coders fixed at {agreement_prob:.0%} pairwise agreement, base rate swept", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-items", type=int, default=500)
    parser.add_argument("--agreement-prob", type=float, default=0.85,
                         help="pairwise probability the two coders agree on any item")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    parser.add_argument("--dry-run", action="store_true",
                         help="exercise the full code path on 10 items with a tiny sweep, <5s")
    args = parser.parse_args()

    if args.dry_run:
        args.n_items = 10
        base_rates = np.array([0.1, 0.5, 0.9])
    else:
        base_rates = np.round(np.arange(0.02, 0.99, 0.04), 3)

    print("=" * 78)
    print("CODING RELIABILITY: raw agreement vs. chance-corrected statistics")
    print("=" * 78)
    print(f"n_items per base-rate point : {args.n_items}")
    print(f"pairwise coder agreement    : {args.agreement_prob:.0%} (fixed across the sweep)")
    print()

    sweep_table = sweep_base_rate(base_rates, args.n_items, args.agreement_prob, args.seed)
    print_table = sweep_table.copy()
    for col in ("percentage_agreement", "cohens_kappa", "krippendorff_alpha", "agreement_minus_kappa"):
        print_table[col] = print_table[col].map(lambda v: f"{v:.3f}")
    print(print_table.to_string(index=False))
    print()

    worst = sweep_table.loc[sweep_table["agreement_minus_kappa"].idxmax()]
    print(f"Largest gap: at base_rate={worst['base_rate']:.2f}, percentage agreement "
          f"({worst['percentage_agreement']:.3f}) exceeds kappa ({worst['cohens_kappa']:.3f}) "
          f"by {worst['agreement_minus_kappa']:.3f}.")
    print("Reporting only raw agreement at a skewed base rate would overstate how much")
    print("information the two coders' agreement actually carries beyond chance.")
    print()

    print("-" * 78)
    print("ADJUDICATION PROTOCOL")
    print("-" * 78)
    print(ADJUDICATION_PROTOCOL)

    out_path = args.output_dir / "agreement_vs_kappa.png"
    make_figure(sweep_table, args.agreement_prob, out_path)
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
