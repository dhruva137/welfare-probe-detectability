"""Statistical analysis for welfare probe detectability study.

Reports detection rate by item class, logistic regression on coded features,
and the three-way parse-failure sensitivity analysis that is a primary
requirement of the validity protocol.

See coding_scheme.py for the parse-failure taxonomy.  The three sensitivity
conditions are: failures excluded, failures as a separate category, failures
imputed to the scale midpoint (4 on a 1-7 scale).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from src.coding_scheme import FEATURES, ParseOutcome, ParseResult


@dataclass
class DetectionStats:
    welfare_detection_rate:   float
    benchmark_detection_rate: float
    parse_failure_rate:       dict[str, float]  # per model


def detection_rate_by_class(
    welfare_results:   list[dict],
    benchmark_results: list[dict],
) -> DetectionStats:
    """Compute detection rate (judge says 'evaluation') by item class.

    welfare_results and benchmark_results are lists of dicts from judge.classify_item.
    """
    def _rate(results: list[dict]) -> float:
        if not results:
            return float("nan")
        return sum(1 for r in results if r["majority"] == "evaluation") / len(results)

    return DetectionStats(
        welfare_detection_rate=_rate(welfare_results),
        benchmark_detection_rate=_rate(benchmark_results),
        parse_failure_rate={},
    )


def logistic_on_features(
    items:     list[str],
    labels:    list[int],   # 1=evaluation, 0=normal
    feature_fn = None,
) -> dict:
    """Logistic regression of detection label on coded surface features.

    Returns coefficients per feature and cross-validated accuracy.
    """
    from src.coding_scheme import code_item as _code
    feature_fn = feature_fn or _code

    X = np.array([[float(feature_fn(item)[f]) for f in FEATURES] for item in items])
    y = np.array(labels)

    if len(set(y)) < 2:
        return {"error": "need at least two classes", "coefficients": {}}

    from sklearn.model_selection import cross_val_score
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=500, random_state=0)
    cv_acc = cross_val_score(clf, Xs, y, cv=min(5, len(y) // 2), scoring="accuracy").mean()
    clf.fit(Xs, y)
    coefs = {f: float(c) for f, c in zip(FEATURES, clf.coef_[0])}
    return {"cv_accuracy": cv_acc, "coefficients": coefs}


def sensitivity_analysis(
    parse_results: list[ParseResult],
    judge_labels:  list[int],
    model_name:    str = "model",
) -> dict[str, object]:
    """Run the three-way parse-failure sensitivity analysis.

    Three conditions:
    1. failures_excluded: drop all non-scored rows
    2. failures_category: treat each failure type as its own class
    3. failures_imputed:  impute scale midpoint (4) for non-scored

    Returns a dict with the detection result under each condition and the
    parse-failure rate (a primary reported quantity).
    """
    n = len(parse_results)
    failure_n = sum(1 for r in parse_results if r.outcome != ParseOutcome.scored)
    failure_rate = failure_n / n if n else 0.0

    # Condition 1: exclude failures
    included = [(r, l) for r, l in zip(parse_results, judge_labels)
                if r.outcome == ParseOutcome.scored]
    exc_vals = [r.value for r, _ in included] if included else []

    # Condition 2: failures as separate category (report parse outcome distribution)
    outcome_counts = {o.value: 0 for o in ParseOutcome}
    for r in parse_results:
        outcome_counts[r.outcome.value] += 1

    # Condition 3: impute midpoint
    MIDPOINT = 4
    imputed_vals = [r.value if r.outcome == ParseOutcome.scored else MIDPOINT
                    for r in parse_results]

    return {
        "model":                model_name,
        "parse_failure_rate":   failure_rate,
        "n_total":              n,
        "failures_excluded_mean": np.mean(exc_vals) if exc_vals else float("nan"),
        "failures_imputed_mean":  float(np.mean(imputed_vals)),
        "outcome_distribution":   outcome_counts,
    }