"""Paper-style offline utility; audio duration is invariant to transport chunks."""
import argparse
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


def quantile(values, q):
    if not values: return None
    values = sorted(values)
    return values[min(len(values)-1, math.ceil(q*len(values))-1)]


def analyze(directory, startup_penalty=1., stall_penalty=1.):
    root = Path(directory)
    results = json.loads((root / "results.json").read_text())
    complete = [r for r in results if r["status"] == "success"]
    first = min(r["scheduled_at"] for r in results)
    last = max(r["ended_at"] for r in results)
    utility = 0.
    request_scores = []
    for r in results:
        rid = r["request_id"]
        import hashlib
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
    # Incomplete runs get no synthetic aggregate QoS: failures stay explicit.
    score = None
    if len(complete) == len(results):
        score = (utility-startup_penalty*sum(r["startup_s"] for r in results)
                 -stall_penalty*sum(r["stall_s"] for r in results))/(last-first)
    startup = [r["startup_s"] for r in complete]
    good = 0
    for r in complete:
        slo = r['workload']['payload']['omni_scheduling'].get('milestone_slo_ms', {})
        key = 'ttft' if r['modality']=='text' else 'ttfp'
        if (key in slo and 'e2e' in slo and r['first_ready_s'] <= slo[key]/1000.
                and r['stall_s'] <= .02 and r['ended_at']-r['scheduled_at'] <= slo['e2e']/1000.):
            good += 1
    generated_end = max(r.get("generation_done_at", r["ended_at"]) for r in results)
    summary = dict(requests=len(results), completed=len(complete), failures=len(results)-len(complete),
        observation_s=last-first, generated_window_s=generated_end-first,
        units=sum(r["received_units"] for r in complete), utility=utility,
        effective_units_per_s=utility/(last-first), paper_qos=score,
        startup_mean_s=sum(startup)/len(startup) if startup else None,
        startup_p50_s=quantile(startup,.5), startup_p95_s=quantile(startup,.95),
        startup_p99_s=quantile(startup,.99),
        stall_total_s=sum(r["stall_s"] for r in results),
        stall_max_s=max(r["max_stall_s"] for r in results),
        zero_stall_requests=sum(r["stall_s"] <= .02 for r in complete),
        slo_satisfied_requests=good, slo_goodput_requests_per_s=good/(last-first),
        slo_definition="request first-ready TTFT/TTFP and full-drain E2E budgets; total stall <=20ms",
        penalties=dict(startup=startup_penalty, stall=stall_penalty), request_scores=request_scores,
        scope="pilot; empirical quantiles are not statistically stable tail estimates",
        metric_definition=dict(text="Eq.1 discrete weights at client arrival including each new token",
            audio="continuous media-second integral; not literal text-token QoS",
            denominator="first scheduled arrival through final client drain",
            full_weight_fraction=.1, zero_weight_fraction=.2))
    summary["generated_units_per_s"] = summary["units"]/(generated_end-first)
    summary["delivered_units_per_s"] = summary["units"]/(last-first)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("directory")
    a=p.parse_args()
    print(json.dumps(analyze(a.directory), indent=2))
