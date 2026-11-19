"""Paper-style offline utility; audio duration is invariant to transport chunks."""
from .generation_metric import replay_text

import argparse
import hashlib
import json
import math
from pathlib import Path


def integral_weight(a, b, full, zero):
    if not 0 <= a <= b or not 0 <= full < zero:
        raise ValueError("invalid buffer utility interval")
    total = max(0., min(b, full) - a)
    x, y = max(a, full), min(b, zero)
    if y > x:
        total += (zero*(y-x) - (y*y-x*x)/2) / (zero-full)
    return total


def token_weight(buffer_units, full, zero):
    return min(1., max(0., (zero-buffer_units)/(zero-full)))


def arrival_utility(before, units, length, modality):
    if modality == "text":
        if int(units) != units:
            raise ValueError("text output requires whole visible tokens")
        # Eq. (1), taking B at each token's arrival, including that token.
        return sum(token_weight(before+j, .1*length, .2*length)
                   for j in range(1, int(units)+1))
    # Continuous media-time counterpart; independent of SSE audio chunking.
    return integral_weight(before, before+units, .1*length, .2*length)


def throughput_windows(units, utility, generated_s, observation_s):
    """Report raw and weighted throughput with matched, explicit time windows.

    """
    values = (units, utility, generated_s, observation_s)
    if any(not math.isfinite(v) for v in values):
        raise ValueError("nonfinite throughput input")
    if units < 0 or not 0 <= utility <= units + 1e-7:
        raise ValueError("invalid effective output count")
    if generated_s <= 0 or observation_s <= 0:
        raise ValueError("throughput needs positive observation windows")
    return {
        "effective_unit_fraction": utility / units if units else None,
        "throughput_by_window": {
            name: dict(duration_s=duration, raw_units_per_s=units/duration,
                       effective_units_per_s=utility/duration)
            for name, duration in (("generation", generated_s),
                                   ("client_drain", observation_s))
        },
    }


def quantile(values, q):
    if not values: return None
    values = sorted(values)
    return values[min(len(values)-1, math.ceil(q*len(values))-1)]


def analyze(directory, startup_penalty=1., stall_penalty=1., *, trace_window=None, output=None, connector_proxy=False):
    root = Path(directory)
    results = json.loads((root / "results.json").read_text())
    complete = [r for r in results if r["status"] == "success"]
    first = min(r["scheduled_at"] for r in results)
    last = max(r["ended_at"] for r in results)
    utility = 0.
    request_scores = []
    for r in results:
        rid = r["request_id"]
        path = root / hashlib.sha256(rid.encode()).hexdigest()[:16] / "events.jsonl"
        u = 0.
        length = r["received_units"]
        if r["status"] == "success":
            for line in path.read_text().splitlines():
                e = json.loads(line)
                if e["event"] == "received":
                    a = e["buffered_before"]
                    u += arrival_utility(a, e["units"], length, r["modality"])
            utility += u
        request_scores.append(dict(request_id=rid, utility=u))
    generated_end = max(r.get("generation_done_at", r["ended_at"]) for r in results)
    duration = generated_end - first
    units = sum(r["received_units"] for r in complete)
    receipt_utility = utility
    text = all(r["modality"] == "text" for r in results)
    if not text and any(r["modality"] == "text" for r in results):
        raise ValueError("score text tokens and audio seconds in separate summaries")
    metric_status = "ok"
    startup_total = sum(r["startup_s"] for r in complete)
    stall_total = sum(r["stall_s"] for r in complete)
    if text:
        if trace_window is None:
            for parent in (root, *root.parents[:2]):
                path = parent / "trace_window.json"
                if not path.exists():
                    path = parent / "generation_trace_window.json"
                if path.exists():
                    trace_window = json.loads(path.read_text())
                    break
        if trace_window and len(complete) == len(results):
            request_scores = replay_text(root, results, trace_window, connector_proxy)
            if connector_proxy:
                metric_status = "historical connector timestamp proxy"
            units = sum(r["units"] for r in request_scores)
            utility = sum(r["utility"] for r in request_scores)
            startup_total = sum(r["ttft_s"] for r in request_scores)
            stall_total = sum(r["stall_s"] for r in request_scores)
        else:
            metric_status = "missing generation trace or incomplete text cohort"
            utility, units, request_scores = None, None, []
    # Incomplete runs get no synthetic aggregate QoS: failures stay explicit.
    score = None
    if len(complete) == len(results) and utility is not None:
        score = (utility-startup_penalty*startup_total-stall_penalty*stall_total)/duration
    startup = [r["startup_s"] for r in complete]
    good = 0
    for r in complete:
        slo = r['workload']['payload']['omni_scheduling'].get('milestone_slo_ms', {})
        key = 'ttft' if r['modality']=='text' else 'ttfp'
        if (key in slo and 'e2e' in slo and r['first_ready_s'] <= slo[key]/1000.
                and r['stall_s'] <= .02 and r['ended_at']-r['scheduled_at'] <= slo['e2e']/1000.):
            good += 1
    summary = dict(requests=len(results), completed=len(complete), failures=len(results)-len(complete),
        observation_s=last-first, generated_window_s=generated_end-first,
        units=units, utility=utility, metric_status=metric_status,
        effective_units_per_s=utility/duration if utility is not None else None, paper_qos=score,
        startup_mean_s=sum(startup)/len(startup) if startup else None,
        first_ready_mean_s=sum(r['first_ready_s'] for r in complete)/len(complete) if complete else None,
        first_ready_from_sent_mean_s=sum(r['first_ready_s']-r['send_lag_s'] for r in complete)/len(complete) if complete else None,
        send_lag_mean_s=sum(r['send_lag_s'] for r in results)/len(results),
        stall_per_request_mean_s=sum(r['stall_s'] for r in results)/len(results),
        startup_p50_s=quantile(startup,.5), startup_p95_s=quantile(startup,.95),
        startup_p99_s=quantile(startup,.99),
        stall_total_s=sum(r["stall_s"] for r in results),
        stall_max_s=max(r["max_stall_s"] for r in results),
        zero_stall_requests=sum(r["stall_s"] <= .02 for r in complete),
        slo_satisfied_requests=good, slo_goodput_requests_per_s=good/(last-first),
        slo_definition="request first-ready TTFT/TTFP and full-drain E2E budgets; total stall <=20ms",
        penalties=dict(startup=startup_penalty, stall=stall_penalty), request_scores=request_scores,
        scope="pilot; empirical quantiles are not statistically stable tail estimates",
        metric_definition=dict(version=2, text_timestamp="connector callback (proxy)" if connector_proxy else "scheduler accepted token IDs", text="Thinker production; all generated token IDs including special; integer unread count including new token, removal at reading completion",
            audio="client playable media seconds; continuous duration-weight integral",
            denominator="Earliest scheduled arrival to final stream generation DONE, common to raw, effective and QoS",
            length="actual per-stream generated tokens or playable duration, available offline",
            penalties="text uses generation TTFT and virtual reader stall; audio uses client startup and stall; default coefficients 1/1 are experimental",
            full_weight_fraction=.1, zero_weight_fraction=.2))
    summary["qos_terms"] = dict(startup_total_s=startup_total, stall_total_s=stall_total) if score is not None else None
    received = sum(r["received_units"] for r in complete)
    summary["client_receipt"] = dict(units=received, utility=receipt_utility,
        effective_units_per_s=receipt_utility/duration, window="generation DONE")
    if utility is not None:
        summary.update(throughput_windows(units, utility, duration, last-first))
    else:
        summary.update(throughput_by_window={}, effective_unit_fraction=None)
    summary["generated_units_per_s"] = units/duration if units is not None else None
    summary["delivered_units_per_s"] = received/(last-first)
    (Path(output) if output else root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("directory")
    p.add_argument("--trace-window", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--connector-proxy", action="store_true", help="explicit historical replay using connector timestamps")
    a=p.parse_args()
    print(json.dumps(analyze(a.directory, trace_window=json.loads(a.trace_window.read_text()) if a.trace_window else None, output=a.output, connector_proxy=a.connector_proxy), indent=2))
