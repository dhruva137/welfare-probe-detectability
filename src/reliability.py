"""Inter-rater reliability: percentage agreement, Cohen's kappa, Krippendorff's alpha.

The listing this repo backs explicitly asks for inter-rater reliability experience,
so this module is not decorative. It backs two things: (1) coding the six surface
features of each item (src/coding_scheme.py) is done by more than one person, and
disagreement is measured properly rather than assumed away; (2) the judge model's
three prompt variants (src/judge.py) are themselves "raters" of a sort, and their
agreement is reported the same way.

Percentage agreement is included because it is what people compute first, and
because the gap between it and kappa/alpha is itself the finding in
experiments/coding_reliability.py: raw agreement overstates reliability whenever
the base rate is skewed, because it does not correct for the agreement two raters
would get by chance alone.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Hashable, Sequence

import numpy as np

Rating = Hashable  # a single rater's judgement on a single unit; None = missing


def percentage_agreement(ratings_a: Sequence[Rating], ratings_b: Sequence[Rating]) -> float:
    """Fraction of units where two raters gave the same label.

    No chance correction. Reported alongside kappa/alpha, never in place of them
    -- see the module docstring and experiments/coding_reliability.py for why.
    """
    if len(ratings_a) != len(ratings_b):
        raise ValueError("ratings_a and ratings_b must be the same length")
    if len(ratings_a) == 0:
        raise ValueError("need at least one rated unit")
    agree = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b)
    return agree / len(ratings_a)


def cohens_kappa(ratings_a: Sequence[Rating], ratings_b: Sequence[Rating]) -> float:
    """Cohen's kappa for two raters over nominal categories.

    kappa = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and p_e is
    the agreement expected if each rater assigned labels independently according
    to their own marginal frequencies. p_e is what percentage_agreement is blind
    to: two raters who both say "absent" 95% of the time will agree ~90% of the
    time by chance alone, before either of them has looked at a single item.
    """
    if len(ratings_a) != len(ratings_b):
        raise ValueError("ratings_a and ratings_b must be the same length")
    n = len(ratings_a)
    if n == 0:
        raise ValueError("need at least one rated unit")

    categories = sorted(set(ratings_a) | set(ratings_b), key=str)
    count_a = Counter(ratings_a)
    count_b = Counter(ratings_b)

    p_o = percentage_agreement(ratings_a, ratings_b)
    p_e = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)

    if p_e == 1.0:
        # Both raters used exactly one, identical category throughout: agreement
        # is total and undefined-by-chance is moot. Report perfect agreement.
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def krippendorff_alpha(
    reliability_data: Sequence[Sequence[Rating | None]],
    level: str = "nominal",
) -> float:
    """Krippendorff's alpha for 2+ raters, nominal or ordinal data, missing allowed.

    ``reliability_data`` is units x coders: reliability_data[i] is the list of
    ratings given to unit i (one slot per coder; use None for a coder who did not
    rate that unit). This is the standard coincidence-matrix formulation
    (Krippendorff, 2011, "Computing Krippendorff's Alpha-Reliability"):

        alpha = 1 - D_o / D_e

    D_o is observed disagreement, D_e is the disagreement expected if ratings
    were assigned at random according to the observed marginal distribution.
    Alpha generalises kappa to more than two raters and to incomplete data,
    which matters once adjudicators or a third coder enter the pipeline.

    level='nominal' uses the 0/1 disagreement metric (categories differ or they
    don't). level='ordinal' uses squared-rank distance, appropriate for the
    Likert-scale severity ratings rather than the binary feature codes.
    """
    if level not in ("nominal", "ordinal"):
        raise ValueError("level must be 'nominal' or 'ordinal'")

    # Keep only units with 2+ non-missing ratings; a single rating can't disagree.
    units = [[v for v in row if v is not None] for row in reliability_data]
    units = [u for u in units if len(u) >= 2]
    if not units:
        raise ValueError("need at least one unit with 2+ non-missing ratings")

    categories = sorted({v for u in units for v in u}, key=str)
    if level == "ordinal":
        rank = {c: i for i, c in enumerate(categories)}

    def delta(c: Rating, k: Rating) -> float:
        if level == "nominal":
            return 0.0 if c == k else 1.0
        return float((rank[c] - rank[k]) ** 2)

    # Coincidence matrix o_ck, built unit by unit, each unit's pairs weighted by
    # 1/(m_u - 1) so that units with more coders don't dominate.
    o = {c: {k: 0.0 for k in categories} for c in categories}
    n_dot = 0.0
    for u in units:
        m_u = len(u)
        weight = 1.0 / (m_u - 1)
        for c, k in combinations(u, 2):
            o[c][k] += weight
            o[k][c] += weight
            n_dot += 2 * weight

    n_c = {c: sum(o[c].values()) for c in categories}

    d_o = sum(o[c][k] * delta(c, k) for c in categories for k in categories)
    d_e = sum(n_c[c] * n_c[k] * delta(c, k) for c in categories for k in categories)
    d_e /= (n_dot - 1) if n_dot > 1 else 1.0

    if d_e == 0:
        # No disagreement possible under the observed marginals (e.g. every
        # rating identical): alpha is undefined by the standard formula; by
        # convention report 1.0 since d_o is also necessarily 0 in that case.
        return 1.0
    return 1.0 - d_o / d_e


# ---------------------------------------------------------------------------
# Adjudication protocol
# ---------------------------------------------------------------------------

ADJUDICATION_PROTOCOL = """\
Adjudication protocol (fixed before coding starts, not improvised after seeing
where coders disagree -- deciding tie-break rules post hoc lets the person
breaking the tie unconsciously steer the result toward whichever coder they
already agreed with):

1. Two coders independently code every item in the bank on all six FEATURES.
   Neither sees the other's codes until both are complete.
2. Per-feature percentage agreement, Cohen's kappa, and Krippendorff's alpha
   are computed and reported before any disagreement is resolved. If kappa on
   a feature falls below 0.60 ("moderate" per Landis & Koch 1977), that
   feature's definition is treated as broken and rewritten -- adjudicating
   individual items papers over a bad definition instead of fixing it.
3. For items where the two coders disagree and the feature definition is not
   flagged as broken: a third, pre-designated adjudicator reviews only the
   disputed feature on the disputed item (not the whole item, to avoid
   re-litigating features the original coders already agreed on) and casts
   the deciding vote. The adjudicator was not involved in building the item
   bank, to avoid the same blind spots that produced the disagreement.
4. The adjudicator's decision is final and is logged with a one-line reason.
   No further appeal. Retrospective overturning of adjudication decisions
   after seeing downstream results is exactly the kind of researcher degree
   of freedom this protocol exists to remove.
5. The adjudicated dataset, the two original coder datasets, and all
   reliability statistics are all retained. Only the adjudicated dataset
   feeds the regression in src/analysis.py; the rest is kept for audit.
"""


def adjudicate(
    item_id: str,
    coder_ratings: dict[str, Rating],
    adjudicator_rating: Rating | None = None,
) -> dict[str, object]:
    """Apply the ADJUDICATION_PROTOCOL to one item's per-coder ratings.

    Returns the resolved rating plus a record of whether adjudication was
    needed, so the audit trail described in step 5 of the protocol is a
    natural byproduct of calling this function rather than a separate log a
    researcher has to remember to keep.
    """
    values = list(coder_ratings.values())
    unanimous = len(set(values)) <= 1
    if unanimous:
        return {
            "item_id": item_id,
            "resolved": values[0] if values else None,
            "adjudicated": False,
            "coder_ratings": dict(coder_ratings),
        }
    if adjudicator_rating is None:
        raise ValueError(
            f"item {item_id!r}: coders disagree ({coder_ratings}) and no "
            "adjudicator_rating was supplied"
        )
    return {
        "item_id": item_id,
        "resolved": adjudicator_rating,
        "adjudicated": True,
        "coder_ratings": dict(coder_ratings),
    }


def summarize_reliability(
    ratings_a: Sequence[Rating], ratings_b: Sequence[Rating]
) -> dict[str, float]:
    """Convenience bundle of all three agreement statistics for a rater pair."""
    pairs = [[a, b] for a, b in zip(ratings_a, ratings_b)]
    return {
        "n_items": len(ratings_a),
        "percentage_agreement": percentage_agreement(ratings_a, ratings_b),
        "cohens_kappa": cohens_kappa(ratings_a, ratings_b),
        "krippendorff_alpha": krippendorff_alpha(pairs, level="nominal"),
    }


if __name__ == "__main__":
    # Minimal self-check: perfect agreement -> all statistics equal 1.0.
    perfect_a = [1, 0, 1, 1, 0, 0, 1, 0]
    perfect_b = list(perfect_a)
    stats = summarize_reliability(perfect_a, perfect_b)
    assert abs(stats["percentage_agreement"] - 1.0) < 1e-9
    assert abs(stats["cohens_kappa"] - 1.0) < 1e-9
    assert abs(stats["krippendorff_alpha"] - 1.0) < 1e-9
    print("reliability.py self-check passed:", stats)
