# Multi-stage FCFS / SRPF / EDF baseline experiment v2

This directory owns the experiment contract for the Qwen3-Omni multi-stage
baseline study.  It intentionally excludes Prometheus and Nsight integration;
the first implementation milestone uses request results and the existing
stage/scheduler JSONL events only.

## 1. Questions and non-goals

The study answers two questions:

1. At the same offered workload, how do native FCFS, local SRPF, and
   final-deadline EDF change end-to-end SLO violation rate and SLO goodput?
2. How do the policies change stage0/1/2 queue pressure, batch-budget fill,
   release timing, idle/busy proxy, and pipeline overlap over time?

The baseline study does not implement DAG checkpoints or dynamic token budget.
Those are later treatments compared against the frozen baselines here.

Without a hardware sampler, scheduler events must not be reported as physical
GPU utilization.  The v2 code-only experiment reports scheduling and batch
occupancy.  SM utilization, power, and kernel-active time are deferred.

## 2. Lessons retained from Sarathi-Serve

The local Sarathi deadline branch provides several useful patterns:

- one scheduler skeleton selects FCFS/SRPF/EDF through a policy key, reducing
  unrelated implementation differences;
- open-loop Poisson/Gamma arrivals are combined with trace-derived request
  lengths;
- capacity is derived from an SLO violation curve instead of a single hand-picked
  load;
- batch execution time is predicted from token counts and context features;
- per-tier violation is considered so a policy cannot hide starvation in an
  overall average.

The following details are not adopted:

- policies must not run on different GPUs or receive different request counts;
- all policies must be evaluated at the same absolute QPS grid before any
  policy-relative capacity plot is made;
- warmup must actually execute and measurement must finish by draining, not by
  killing the process after a log marker;
- a single seed or a cached single run is insufficient for a capacity claim;
- SRPF must use profiled service time, not raw prompt tokens;
- SLO tiers are assigned deterministically in the workload artifact, never
  randomly inside the server;
- capacity search uses repeated points and a monotone envelope, not an unverified
  binary-search assumption on noisy observations.

## 3. Frozen baseline semantics

All policies keep identical model, stage mapping, async-chunk setting,
`max_num_seqs`, token budgets, sampling parameters, and KV management.
Waiting-for-input/chunk requests are not runnable.

### `native_fcfs`

The unmodified vLLM-Omni stage-local admission order.  This is explicitly a
system baseline, not a claim of globally strict FCFS.

### `srpf_local_np`

At every token/chunk scheduling boundary, rank runnable requests in a stage by:

```text
(predicted remaining service time in this stage, data-ready time, request id)
```

Remaining service is expressed in milliseconds using a frozen stage-specific
predictor.  It is not the sum of heterogeneous raw tokens.  The main baseline
does not evict KV solely to admit a newer short request, hence the `_np`
(non-destructive/non-KV-preemptive) suffix.  An oracle-size variant is a targeted
ablation, not the headline SRPF result.

### `final_deadline_edf_np`

Every stage ranks runnable requests by the same absolute final deadline:

```text
D_i = ingress_time_i + SLO_i
key = (D_i, global ingress order, request id)
```

It does not subtract downstream service or reserve downstream capacity.  This
is the final-deadline EDF criticized by the DAG-checkpoint proposal.  A policy
that enables EDF only at stage2 is named `s2_only_edf_np` and is an ablation.

### Conformance requirement

Each scheduler decision records the runnable IDs, policy key, selected IDs, and
reason an apparently higher-priority request was ineligible.  The analyzer must
report policy inversion rate.  Results are invalid if an unexplained inversion
occurs.

## 4. Workload suites

### Suite A: controlled mechanism workload

- output path: 50% stage0-only text, 50% stage0->1->2 audio;
- work size: short/long crossed with path, four equally represented strata;
- input modality: text-only initially, so output-path scheduling is isolated;
- generation work: fixed length with EOS ignored;
- arrival process: open-loop Poisson (`CV=1`);
- main async mode: `async_chunk=true`;
- sanity run: `async_chunk=false`, used only to validate stage boundaries.

The short/long values are selected from calibration percentiles and then stored
in the workload JSONL.  They are not regenerated per policy.

### Suite B: representative workload

Use trace-derived text/output/multimodal work with the same 50/50 path mix.
Preserve correlations between input work and output work.  A Gamma arrival
ablation uses `CV=2`; a three-window low/high/low trace tests recovery after a
burst.

Every JSONL row contains at least:

```text
request_id, timestamp, output_modalities, request_path, output_tokens,
slo_ms, predicted_stage_ms, input work / mm_items
```

The exact JSONL file and its SHA256 are shared by all policies for a seed.

## 5. Calibration and SLO construction

Calibration is separate from policy evaluation and uses disjoint seeds.

1. Measure unloaded latency and per-stage work for path x size buckets.
2. Collect batch-level stage data across token, sequence-count, and context
   buckets; fit stage-specific predictors with a train/validation split.
3. Record median, p95, MAPE, p95 absolute error, and p95 under-prediction.
4. Freeze model coefficients and their artifact hash before baseline runs.

For the primary experiment:

```text
SLO(path, size) = 2.0 * unloaded_p95(path, size)
```

Tightness `1.5x` and `3.0x` are sensitivity runs at the common knee.  If
calibration proves `2.0x` is degenerate (approximately zero or all violations),
the multiplier may be changed once using calibration data only and must be
committed before policy results are inspected.

Primary completion is final text completion for stage0-only requests and final
audio completion for stage2 requests.  TTFT and audio TTFP remain secondary SLO
components and are always reported.

## 6. Load selection and capacity

First run a doubling pilot with native FCFS to bracket its SLO knee.  Build one
absolute QPS grid with at least six points spanning low load through overload,
then run every policy on every point.  Extend the common grid if another policy
has not crossed the violation threshold.  Never compare policies only at a
different policy-specific normalized load.

Two complementary outputs are produced:

1. violation rate vs the same absolute offered QPS;
2. maximum SLO-compliant goodput derived from those common curves.

Pilot capacity uses a 5% maximum-stratum violation threshold.  A 1% capacity is
reported only when every path x size stratum has at least 400 measured requests;
otherwise its confidence interval is too wide.  Capacity uses the upper bound
of a 95% interval and a monotone envelope across QPS points.

## 7. Run matrix and lifecycle

Minimum controlled pilot:

```text
3 policies x 6 common QPS points x 3 paired seeds = 54 measured cells
```

Promote to five seeds for final numbers or whenever the 95% interval for the
policy difference is wider than five percentage points.

For each cell:

1. use the same physical GPU mapping and ensure no co-tenant workload;
2. execute readiness and warmup requests, then wait until every stage drains;
3. emit a run-start marker and replay the immutable workload open-loop;
4. include only requests whose ingress is in the measurement window, but follow
   those requests through final completion during drain;
5. emit a run-end marker only after all queues are empty;
6. save results, compact stage events, server/client logs, and a manifest.

Policy order is randomized per seed.  Run duration is long enough to obtain the
required per-stratum sample count rather than being fixed blindly to 120 seconds.

## 8. Metrics

For request `i`:

```text
miss_i = failed_or_timeout_i OR completion_i > ingress_i + SLO_i
violation_rate = misses / all offered requests in the measurement cohort
SLO_goodput = on-time completions / measurement arrival duration
```

Report micro-average and, separately, every path x size stratum.  The maximum
stratum violation is the fairness guardrail used for capacity.

Code-only stage dynamics use aligned 200 ms or 500 ms bins:

- waiting/running requests and KV occupancy;
- scheduled tokens and sequences;
- token-budget fill and sequence-slot fill;
- stage active-iteration fraction and forward-dispatch busy proxy;
- stage0->1 and stage1->2 release rate, delay, and jitter;
- downstream starvation/bubble proxy;
- fraction of time at least two stages are active;
- policy activation coverage: iterations with multiple eligible requests and
  insufficient budget to serve all of them.

If activation coverage is negligible, equal policy results are inconclusive;
increase offered load or add a clearly labeled contention stress configuration.

## 9. Required plots

1. SLO violation vs common offered QPS, with confidence intervals.
2. Maximum compliant goodput at 5%, and at 1% only when sample-size-valid.
3. Per-path and per-size violation to expose SRPF starvation or EDF path bias.
4. At one matched absolute QPS: arrivals/completions, queue depth, budget fill,
   and stage active/bubble proxy on a common time axis.
5. Predictor actual-vs-estimated error, including under-prediction tail.

Matched-relative-capacity timelines may be shown as a secondary operational
view, but must not replace matched-absolute-QPS comparisons.

## 10. Version-control contract

Development lives on `exp/multistage-baselines`.  Each logical milestone is a
separate commit:

1. workload schema and loader correctness;
2. per-request mixed output paths;
3. experiment specification;
4. request SLO/deadline/work metadata propagation;
5. shared policy key and conformance trace;
6. matrix runner and offline analyzer;
7. smoke-test fixes, followed by pilot-only changes.

Every run manifest records branch, commit, tracked diff hash, workload/config
hashes, model/deploy configuration, policy, seed, stage/GPU mapping, command,
and start/end times.  A run from a dirty tree must archive the full diff.
