# Cross-stage terminal output barrier

The final backend stage can finish before the orchestrator consumes earlier
stages' queued outputs. Cleaning up on that final-stage envelope alone loses
upstream text and closes the frontend stream too soon.

The orchestrator now records processed terminal envelopes per request. Normal
completion requires the final stage to be terminal, all submitted stages to be
terminal, and no pending restart stage. Real output chunks are forwarded
immediately. An early final-stage chunk has its **queue-level** finished flag
cleared; the underlying stage output and its metrics are unchanged. Once all
required terminals have been routed, a FIFO `request_done` control message ends
the request. It carries no generated text/audio and is not counted as output or
as another metrics sample. Both sync and async entrypoints share the handler.

When the final stage is processed last (including text-only requests), its
original output envelope still carries completion; no extra control message is
needed. Request state and upstream output-processor state remain alive until
the barrier is satisfied. This guarantees enqueue order, not remote delivery
acknowledgment. Errors and cancellation still abort immediately. Existing
execution-version filtering remains in force; a full restart clears terminal
observations, and local recovery clears only affected stages.

Cost: a small per-request set and a stage-count-bounded check, plus one control
message only for deferred completion. Overall completion and resource-binding
release may be later than the final audio block. This reflects actual upstream
draining; it must not be hidden by dropping output. A stage that never emits its
terminal can keep a request pending until cancellation/error; do not introduce
a timeout that silently treats incomplete output as success.

Targeted regression: 115 tests passed across terminal barrier, full/local
recovery, snapshot queue, drain, and execution ledger suites. The barrier tests
cover all six three-stage terminal orders, late text, immediate real audio,
single-stage completion, cancellation, errors, and actual async-consumer
termination without a synthetic chunk. Recovery tests assert that old terminal
observations cannot complete a replacement execution.

Real-load validation is separate: use the unchanged 600-request 80:20 workload,
3.5 req/s, seed 2026091400, 1024/16 output settings, fixed 1:1 admission and
4+1+1 GPUs, isolated client and compact SSE capture. Require no request failures,
no premature-cleanup rejection, full text token/usage and audio count audits,
strict drain, and the unchanged 100ms launch-delay gate. A passing unit suite
alone is not a completed performance experiment.

Broader-suite limitation: `test_orchestrator.py` and
`tests/entrypoints/test_async_omni.py` have 17 pre-existing failures (9 pass),
reproduced by loading the committed pre-change orchestrator/base source into a
fresh Python process. They include raw fixture IDs rejected by the execution
ledger, harness startup timeouts, and missing frontend fixture fields. These
were not bypassed in production. The recovery terminal test was updated to
explicitly model upstream terminals already consumed, as required by the new
completion contract. Ruff is not installed in the container; `git diff --check`
passed. This is not a claim that the entire repository test suite is green.
