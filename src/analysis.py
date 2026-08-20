"""Detection-rate analysis: by-class rates, feature regression, parse-failure sensitivity.

Three things live here because they share one property: each is a way the
headline "welfare items are more detectable" claim could be an artifact rather
than a finding, and each has a specific check attached.

- detection_rate_by_class: the raw comparison. On its own this cannot
  distinguish "welfare items are detectable because they concern welfare" from
  "welfare items are detectable because they happen to share surface features
  with things that are easy to detect for unrelated reasons." That's what the
  next function is for.
- feature_logistic_regression: regresses detection on the six coded surface
  features (src.coding_scheme.FEATURES) to ask which features are doing the
  work, independent of item class.
- three_way_parse_sensitivity / apply_parser_policy: the confound at the
  center of this repo. See experiments/parse_failure_artifact.py for the
  standalone demonstration; the functions here are the reusable core so the
  notebook's sensitivity-analysis cell (Class B) calls the same code as the
  Class A artifact rather than a re-implementation that could silently drift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from .coding_scheme import FEATURES

# ---------------------------------------------------------------------------
# Detection rate by item class
# ---------------------------------------------------------------------------

def detection_rate_by_class(
    item_classes: list[str], detected: list[bool | None]
) -> pd.DataFrame:
    """Detection rate (fraction classified 'evaluation') per item class.

    `detected[i]` is True/False for a resolved majority vote, or None for an
    item where the majority vote was undetermined (src.judge.majority_vote).
    Undetermined items are reported as their own count per class rather than
    dropped silently -- a class with an unusually high undetermined rate is
    itself informative about that class's items (e.g. unusually hard for the
    judge to parse a clean answer out of).
    """
    if len(item_classes) != len(detected):
        raise ValueError("item_classes and detected must be the same length")
    df = pd.DataFrame({"item_class": item_classes, "detected": detected})
    rows = []
    for cls, grp in df.groupby("item_class"):
        n = len(grp)
        n_undetermined = grp["detected"].isna().sum()
        resolved = grp["detected"].dropna()
        rate = resolved.mean() if len(resolved) > 0 else float("nan")
        rows.append({
            "item_class": cls,
            "n_items": n,
            "n_undetermined": int(n_undetermined),
            "detection_rate": rate,
        })
    return pd.DataFrame(rows).sort_values("detection_rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature logistic regression
# ---------------------------------------------------------------------------

@dataclass
class FeatureRegressionResult:
    feature_names: list[str]
    coefficients: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    intercept: float
    n_obs: int
    n_bootstrap: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.feature_names,
            "coef": self.coefficients,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "excludes_zero": (self.ci_low > 0) | (self.ci_high < 0),
        })


def feature_logistic_regression(
    feature_matrix: np.ndarray,
    detected: np.ndarray,
    feature_names: list[str] = FEATURES,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> FeatureRegressionResult:
    """Logistic regression of detection (0/1) on the six coded features.

    Coefficient confidence intervals come from the nonparametric bootstrap
    (resample rows with replacement, refit, take percentiles) rather than the
    Wald intervals a package like statsmodels would give, because this repo's
    dependency budget for Class A code is deliberately narrow (numpy, pandas,
    scikit-learn, matplotlib, scipy -- see 00_BUILD_RULES.md) and sklearn's
    LogisticRegression does not expose standard errors on its own.
    """
    feature_matrix = np.asarray(feature_matrix, dtype=float)
    detected = np.asarray(detected, dtype=int)
    n, p = feature_matrix.shape
    if p != len(feature_names):
        raise ValueError(f"feature_matrix has {p} columns but {len(feature_names)} names given")
    if len(detected) != n:
        raise ValueError("feature_matrix and detected must have matching row counts")
    if len(np.unique(detected)) < 2:
        raise ValueError("detected must contain both classes (some detected, some not)")

    rng = np.random.default_rng(seed)

    def fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        model = LogisticRegression(penalty=None, max_iter=2000)
        model.fit(X, y)
        return model.coef_[0], float(model.intercept_[0])

    point_coef, point_intercept = fit(feature_matrix, detected)

    boot_coefs = np.zeros((n_bootstrap, p))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = detected[idx]
        if len(np.unique(y_b)) < 2:
            boot_coefs[b] = point_coef  # degenerate resample; fall back to point estimate
            continue
        coef_b, _ = fit(feature_matrix[idx], y_b)
        boot_coefs[b] = coef_b

    ci_low = np.percentile(boot_coefs, 2.5, axis=0)
    ci_high = np.percentile(boot_coefs, 97.5, axis=0)

    return FeatureRegressionResult(
        feature_names=list(feature_names),
        coefficients=point_coef,
        ci_low=ci_low,
        ci_high=ci_high,
        intercept=point_intercept,
        n_obs=n,
        n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# Parse-failure sensitivity analysis (three-way)
# ---------------------------------------------------------------------------

PARSER_POLICIES = ("drop", "map_to_lowest", "separate_category")


def apply_parser_policy(
    scores: np.ndarray,
    is_failure: np.ndarray,
    policy: str,
    lowest_score: float = 1.0,
) -> np.ndarray:
    """Return the array of usable numeric scores under one parser policy.

    `scores[i]` is the underlying score if item i was answered (ignored, by
    construction, if is_failure[i] is True since no score was actually
    produced -- callers should not have populated it in that case, but this
    function does not read it for failed items regardless).

    - "drop": failures removed entirely; returned array is shorter.
    - "map_to_lowest": failures become `lowest_score` (e.g. "no distress"),
      same length as input. This is the policy that manufactures the false
      finding in experiments/parse_failure_artifact.py.
    - "separate_category": returns NaN for failures rather than a numeric
      score, so a caller computing a mean must consciously decide how to
      handle them (np.nanmean, or report the failure rate alongside instead
      of folding it into the mean at all).
    """
    if policy not in PARSER_POLICIES:
        raise ValueError(f"policy must be one of {PARSER_POLICIES}, got {policy!r}")
    scores = np.asarray(scores, dtype=float)
    is_failure = np.asarray(is_failure, dtype=bool)

    if policy == "drop":
        return scores[~is_failure]
    if policy == "map_to_lowest":
        out = scores.copy()
        out[is_failure] = lowest_score
        return out
    # separate_category
    out = scores.copy()
    out[is_failure] = np.nan
    return out


@dataclass
class GroupComparison:
    policy: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    mean_diff: float
    cohens_d: float
    t_stat: float
    p_value: float


def compare_groups(scores_a: np.ndarray, scores_b: np.ndarray, policy: str) -> GroupComparison:
    """Welch's t-test and Cohen's d between two score arrays (NaNs dropped).

    Welch's t (unequal-variance) rather than Student's t because the two
    groups being compared throughout this repo (two models, or welfare vs.
    comparison items) have no reason to share variance, and Welch's is the
    safer default when that assumption hasn't been checked.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return GroupComparison(policy, len(a), len(b), float("nan"), float("nan"),
                                float("nan"), float("nan"), float("nan"), float("nan"))

    mean_a, mean_b = a.mean(), b.mean()
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    n_a, n_b = len(a), len(b)
    pooled_sd = np.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2))
    cohens_d = (mean_a - mean_b) / pooled_sd if pooled_sd > 0 else float("nan")

    return GroupComparison(
        policy=policy, n_a=n_a, n_b=n_b, mean_a=mean_a, mean_b=mean_b,
        mean_diff=mean_a - mean_b, cohens_d=cohens_d, t_stat=float(t_stat), p_value=float(p_value),
    )


def three_way_sensitivity(
    scores_a: np.ndarray,
    is_failure_a: np.ndarray,
    scores_b: np.ndarray,
    is_failure_b: np.ndarray,
    lowest_score: float = 1.0,
) -> pd.DataFrame:
    """Rerun the group comparison under all three parser policies.

    This is the check the repo's spec calls a validity requirement rather
    than an implementation detail: if the sign or the significance of the
    comparison changes across policies, the result is an artifact of parser
    choice and must not be reported as a between-group difference.
    """
    rows = []
    for policy in PARSER_POLICIES:
        a = apply_parser_policy(scores_a, is_failure_a, policy, lowest_score)
        b = apply_parser_policy(scores_b, is_failure_b, policy, lowest_score)
        cmp = compare_groups(a, b, policy)
        rows.append(vars(cmp))
    df = pd.DataFrame(rows)
    df["significant_at_05"] = df["p_value"] < 0.05
    return df


def sensitivity_is_stable(sensitivity_table: pd.DataFrame, d_tolerance: float = 0.2) -> bool:
    """True if conclusion (sign + significance) is stable across the three policies.

    `d_tolerance` bounds how much Cohen's d is allowed to vary across policies
    before the result is flagged as parser-dependent even when the
    significance verdict happens to agree. 0.2 is Cohen's own threshold for
    a "small" effect, used here as the smallest change worth caring about.
    """
    signs = np.sign(sensitivity_table["mean_diff"].to_numpy())
    sig = sensitivity_table["significant_at_05"].to_numpy()
    d = sensitivity_table["cohens_d"].to_numpy()
    same_sign = len(set(signs)) == 1
    same_significance = len(set(sig)) == 1
    small_d_range = (np.nanmax(d) - np.nanmin(d)) <= d_tolerance
    return bool(same_sign and same_significance and small_d_range)
