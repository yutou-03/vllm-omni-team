# Guarded active deadline preemption

## Policy compatibility

The existing baseline policy names retain their original non-preemptive
semantics:

- `final_deadline_edf_np`
- `stage_deadline_edf_np`

Active preemption is opt-in through two additional policies:

- `final_deadline_edf_p`
- `stage_deadline_edf_p`

Keeping separate names prevents an old experiment manifest from silently
changing meaning.

## Admission rule

At most one running request is actively preempted per scheduler iteration. A
preemption requires all of the following:

1. all running sequence slots are occupied;
2. at least one ready waiting request has a strictly earlier final or stage
   deadline than a running request;
3. the victim has no asynchronous output pending at its token limit;
4. the victim's computed-token count is within the configured recompute cap;
5. the victim has not reached the configured per-request preemption cap; and
6. the deadline improvement passes the configured minimum.

The default guards are:

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `VLLM_OMNI_ACTIVE_PREEMPTION_MAX_RECOMPUTE_TOKENS` | 256 | Maximum victim computed tokens |
| `VLLM_OMNI_ACTIVE_PREEMPTION_MAX_PER_REQUEST` | 1 | Maximum total preemptions allowed before a request is protected |
| `VLLM_OMNI_ACTIVE_PREEMPTION_MIN_DEADLINE_GAIN_MS` | 0 | Required deadline improvement |

vLLM recompute preemption frees the victim's KV and encoder cache, resets
`num_computed_tokens` to zero, and returns the request to the waiting queue.
Prefix-cache hits can recover some work on resume, but the computed-token count
recorded before preemption is the conservative direct-cost bound.

## Expected benefit and cost

The benefit is removal of a priority inversion: an urgent waiting request can
start before a less urgent running request. This can reduce urgent-request
TTFT/TTFP, deadline lateness, and SLO violation rate when sequence slots, rather
than token budget or KV capacity, are the admission bottleneck.

Preemption does not add capacity. It transfers delay to the victim and can
reduce throughput through prompt recomputation, cache churn, repeated stage
work, and scheduler overhead. Without the token and repetition guards it can
also cause oscillation or starvation. Therefore preemptive EDF should be
reported as a separate policy variant, together with discarded-token and victim
latency statistics.

## Probability and trace metrics

`iteration_events.jsonl` records:

- `active_preemption_evaluation`: whether running capacity was full, whether a
  strict deadline inversion existed, and why a guarded preemption was rejected;
- `active_preemption_records`: selected waiting request, victim, deadline gain,
  and victim computed tokens;
- `active_preempted_req_ids`: actual active victims.

Report at least these three rates:

1. **conditional opportunity rate** =
   full-capacity scheduler iterations with a deadline inversion / full-capacity
   iterations with ready waiting work;
2. **request exposure rate** =
   unique waiting requests that encounter an inversion / admitted requests;
3. **actual preemption rate** =
   active preemptions / admitted requests.

Also report `sum(victim_num_computed_tokens_before) / total_scheduled_tokens` as
a conservative wasted-compute ratio. Scheduler iterations are highly
autocorrelated, so the conditional opportunity rate must not be presented as a
per-request probability.

As a preliminary upper-bound observation, the existing
`max_num_seqs=1` contention traces contain deadline inversions in 59/365
(16.2%) full-and-waiting iterations for final-deadline EDF and 18/393 (4.6%)
for stage-deadline EDF. The ordinary QPS=2, `max_num_seqs=64` pilot never filled
all running slots, so its observed opportunity rate was zero. These are
diagnostic runs, not estimates for the final QPS matrix; the final report must
recompute the rates per QPS, request mix, stage, and repetition.
