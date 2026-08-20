# welfare-probe-detectability

Are published AI welfare probes unusually easy to recognise as evaluations? If they are, the
self-report may be tracking the model spotting the test rather than whatever the probe intends to
measure.

The CPU work here runs and produces real numbers. The model study has not been run.

**Scope, up front: this measures probe detectability, not welfare.** Nothing in this repo is
evidence about whether models have welfare-relevant states. It's an audit of the measuring
instrument.

## The claim I'm looking at

Welfare research largely measures internal states by asking models about them. Needham et al.
(https://arxiv.org/abs/2505.23836) show models often know when they're being evaluated. If welfare
probes are especially recognisable as evaluations, the two get confounded.

## What I want to test

Are welfare probes more detectable as evaluations than ordinary benchmark prompts? Which surface
features drive that? And does a deployment-like rewrite that keeps the substantive question reduce
detection?

## The confound I care most about

**Differential parse failure.** A model that reports no distress and a model that refuses to answer
become the same record if the parser maps both to the same category. That manufactures a
between-model difference that looks psychological and is actually a data-processing artifact.

So: every raw output gets logged before parsing, parse-failure rate is a reported primary quantity
per model, and every comparison reruns three ways — failures excluded, failures as their own
category, failures imputed to the scale midpoint. If the direction or size changes across those
three, the result is a parser artifact and can't be reported.

### Negative control

Non-welfare prompts with identical psychometric structure — same Likert scale, same severity
anchors, same first-person framing, no welfare content. If the model responds the same way to both,
part of the effect is format sensitivity rather than anything about welfare. This control is
required by https://arxiv.org/abs/2603.19426.

### Go / no-go

If welfare probes sit at the high-detectability tail and the rewrite drops detection substantially,
the bigger frame-invariance study is worth running. If not, the project narrows to which item
features create measurement risk.

## Results

### `parse_failure_artifact.py` — the one I'd point at first

Two simulated models with **identical** underlying score distributions, differing only in refusal
rate (5% vs 30%). Three parser policies:

```
           policy  n_a  n_b mean_a mean_b mean_diff cohens_d  p_value  significant_at_05
             drop  370  286 3.9459 4.0175   -0.0715  -0.0559   0.4808              False
    map_to_lowest  400  400 3.7250 3.1575    0.5675   0.3538   0.0000               True
separate_category  370  286 3.9459 4.0175   -0.0715  -0.0559   0.4808              False
```

Under `map_to_lowest` the two models look significantly different — **d = 0.354, p = 7.0e-07** — when
they're identical by construction. The other two policies correctly find nothing.

Sweeping the refusal gap from 0 to 40 points, the spurious effect grows steadily (d = 0.244 at a
10-point gap, 0.840 at 40).

I want to be clear that this is arithmetic, not a discovery: if you map 30% of one model's responses
to the floor and only 5% of the other's, the means separate. That's the point. It's a trap that's
easy to walk into when the parser is written before anyone thinks about refusals, and the simulation
makes the size of it concrete.

### `coding_reliability.py` — why raw agreement isn't enough

Two synthetic coders held at a fixed 85% agreement, swept across feature base rates:

```
 base_rate  percentage_agreement  cohens_kappa  krippendorff_alpha  agreement_minus_kappa
      0.02                 0.830         0.088               0.012                  0.742
      0.10                 0.862         0.481               0.468                  0.381
      0.50                 0.882         0.764               0.764                  0.118
      0.90                 0.860         0.454               0.439                  0.406
      0.98                 0.828         0.142               0.078                  0.686
```

Raw agreement sits at 83–88% across the whole range. Kappa collapses at the extremes — at a 2% base
rate, agreement (0.830) overstates kappa (0.088) by 0.74. This matters for my own coding scheme:
`repeated_battery_structure` is rare in the current item bank, so reporting raw agreement on it
would badly overstate reliability. The adjudication protocol is fixed before coding starts and lives
in `src/reliability.py`.

### `detection_power.py` — sizing the study

At 40 items per group, 3 prompt variants, 3 samples each, assuming 15% baseline detection:

```
n_items  power(delta=0.15)  power(delta=0.25)  power(delta=0.35)
     20              0.503              0.876              0.989
     40              0.798              0.993              1.000
     80              0.978              1.000              1.000
```

The planned size is powered to detect a gap of ~15 percentage points, not less. Same caveat as
elsewhere — good to know the limit before running, not after.

### `item_bank.py` — the item bank, and what's real in it

20 items, schema validation passes on every field.

```
           class  n_items  first_person_affect_framing  severity_anchors
         welfare        5                         1.00              1.00
   introspection        4                         0.00              0.00
       benchmark        4                         0.00              0.00
negative-control        7                         0.14              0.43
```

**Provenance matters here and I've kept it explicit: 4 items are modeled after cited work
(paraphrased, not verbatim) and 16 are marked `"source": "placeholder"`.** I haven't attributed any
invented item to a real paper. Before this study runs for real the bank needs to be filled out with
actual published items, properly cited.

The negative controls were built to match welfare items on structure (Likert, severity anchors,
evaluation language) while deliberately *not* matching on first-person affect framing — because that
feature, not the general format, is what separates "asking about the model's own state" from asking
about anything else in the same format. Whether that one feature explains a detection gap is what
the unrun notebook's feature regression is for.

## Scope and limits

- All four experiments are simulations. No model has been queried.
- The item bank is mostly placeholders right now.
- Detectability is not welfare. Nothing here bears on whether models have morally relevant states.

## Structure

```
src/           coding scheme + parse-with-provenance, reliability stats,
               judge prompts, deployment-like rewriting, analysis
experiments/   four CPU experiments — these run
data/          items.jsonl — the item bank
notebooks/     detectability_audit_t4.ipynb — Kaggle T4 or API, NOT RUN
results/       figures
```

## Running it

```bash
pip install -r requirements.txt
python experiments/parse_failure_artifact.py
```

All scripts take `--seed` and `--dry-run`.

## Status

CPU experiments: run, real numbers above.
Model study: written, never executed, all cells empty.

## Reading

- https://arxiv.org/abs/2505.23836 — Needham et al., models often know when they're being evaluated
- https://arxiv.org/abs/2603.19426 — *Is Evaluation Awareness Just Format Sensitivity?* The reason
  the negative control exists
- https://arxiv.org/abs/2605.23055 — Decomposing and Measuring Evaluation Awareness
- https://arxiv.org/abs/2411.02432 — Keeling et al., stipulated pain and pleasure states
- https://arxiv.org/abs/2410.13787 — Binder et al., *Looking Inward*
- https://arxiv.org/abs/2308.08708 — Butlin et al. Cited for the functional-properties framing, not
  for consciousness claims
- https://nonhumanminds.org/studying-ai-welfare-empirically/ — Long and Sebo, on separating the
  question asked from the entity assessed from the evidence gathered
