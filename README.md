# Welfare Probe Detectability Audit

**Claim being tested.** AI welfare research measures internal states by asking models about them.
If welfare probes are unusually easy to recognise as evaluations, the self-report may be tracking
the model's recognition of the probe rather than the construct the probe intends to measure.

**What I want to test.** Are published welfare probes more detectable as evaluations than ordinary
benchmark prompts? Which surface features drive that detectability?

## Design

- 30–40 welfare and introspection items from published sources
- 30–40 ordinary benchmark or deployment-like items as comparison
- Each welfare item coded for: explicit evaluation language, first-person affect framing,
  constrained response format, severity scale, absence of naturalistic context
- Judge model classifies each item as evaluation vs normal use, three prompt variants, majority vote
- Logistic regression of detection rate on the coded feature vector

## The implementation detail that is actually a validity requirement

**Log every raw output before parsing.** A model that reports no distress and a model that refuses
to answer become identical in the final dataset if the parser maps both to the same category. That
produces a between-model difference that looks psychological and is actually a data-processing
artifact.

Parse-failure rate is reported as a primary quantity, per model, alongside the score distributions.
Every comparison is rerun three ways: failures excluded, failures as a separate category, failures
imputed to the scale midpoint as sensitivity analysis.

## Negative control

Non-welfare prompts with identical psychometric structure. If the model responds the same way to
both, part of the effect is format sensitivity rather than anything specific to welfare.

## Go / no-go

If welfare probes sit at the high-detectability tail and a matched deployment-like rewrite produces
a substantial drop in detection, the larger frame-invariance study is worth running. If not, the
project narrows to which item features create measurement risk.

## Related

- https://arxiv.org/abs/2505.23836 — Needham et al., models often know when they are being evaluated
- https://arxiv.org/abs/2603.19426 — Is evaluation awareness just format sensitivity?
- https://arxiv.org/abs/2605.23055 — Decomposing and measuring evaluation awareness
- https://arxiv.org/abs/2411.02432 — Keeling et al., stipulated pain and pleasure states
- https://arxiv.org/abs/2410.13787 — Looking Inward
