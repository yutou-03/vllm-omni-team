# Stage2 output queue: adjacent status snapshot coalescing

Default Stage2 clients now use SnapshotOutputQueue. Stage0/1 and backend
statistics production are unchanged. No diagnostics or per-message trace I/O
are added to the default queue path.

Only consecutive messages containing known, delta-free SchedulerStats gauges
(running/waiting/skipped counts and KV usage) may replace the queue tail with
the latest snapshot. Engine IDs must match and timestamps must not go backwards.
All data, completion/preemption/control/error messages and nonempty delta/event
statistics form barriers. No coalescing across barriers, no removal of data or
control, and no mutation of previously delivered objects. Unknown schemas fall
back to FIFO. The queue remains unbounded for real data: this is not admission
control or a policy that drops outputs under pressure.

Cache hit/reset deltas, eviction events, speculative decoding/performance stats,
LoRA metadata and DP counters are not treated as replaceable gauges. The latest
snapshot retains its original timestamp and travels through the existing
consumer; this change does not add a new monitoring consumer or promise a
different monitoring sampling rate. Mixed data/statistics messages are untouched.

`VLLM_OMNI_OUTPUT_QUEUE_DIAGNOSTIC` unset or `off` means the production fix.
Explicit `observe` bypasses coalescing to reproduce historical FIFO diagnostics.
Explicit `drop_stats` retains the historical experimental filtering behavior;
it is NOT the production repair and is unsafe as a generic statistics policy.

Tests cover delta/control barriers, future schemas, engine/time ordering,
50,000 snapshots followed by audio, async wakeups and task_done/join accounting,
and default client installation. The previous connector fault-test hang was a
fixture bypassing __init__ without initializing _abandoned_sender_ids: it failed
before reaching the injected callback that stops the loop. Only the test fixture
is corrected; connector production behavior is not changed by this patch.
