"""Report request-level TTFT and client delivery TPOT from frozen V3 traces.

TPOT here is the mean interval between consecutive observed text token arrivals
for one request. The percentiles give every request one vote. A separate pooled
gap percentile detects rare delays hidden by that per-request mean.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1] if ordered else None


def stats(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def summarize(root):
    report = {
        "source": str(root),
        "definition": {
            "ttft": "results.json ttft_s, from scheduled send to first visible text",
            "visible_tpot": "per-request mean consecutive text_token_arrival_monotonic_s gaps; percentiles across requests",
            "gap": "pooled consecutive client-visible text token gaps, including zero gaps for tokens in the same SSE chunk",
            "stage0_tpot": "mean consecutive Stage 0 output-produced events per request, including pauses",
            "percentile": "nearest rank, ceil(p*n)-1",
            "rebuffer": "stream_score.json virtual reading underflow at 4.8 token/s",
        },
        "arms": {},
    }
    for arm in ("nohold16", "buffer16", "dynamic16", "fixed16"):
        path = root / arm
        if not (path / "results.json").exists():
            continue
        requests = json.loads((path / "results.json").read_text())
        score = json.loads((path / "stream_score.json").read_text())
        stage_events = {}
        event_path = root / (arm + "_service") / "traces" / "stage_events.jsonl"
        for line in event_path.open():
            event = json.loads(line)
            if event.get("event") == "stage_study_thinker_output_produced":
                stage_events.setdefault(event["request_id"], []).append(event)
        report["arms"][arm] = {}
        for tier in ("all", "loose", "standard"):
            rows = [r for r in requests if tier == "all" or r["metadata"]["slo_tier"] == tier]
            score_rows = [r for r in score["requests"] if tier == "all" or r["tier"] == tier]
            assert len(rows) == len(score_rows), (arm, tier, len(rows), len(score_rows))
            per_request_tpot = []
            gaps = []
            stage0_tpot = []
            stage0_gaps = []
            for row in rows:
                arrivals = row["text_token_arrival_monotonic_s"]
                assert len(arrivals) == len(row["text_token_ids"]), row["source_request_id"]
                produced = sorted(stage_events[row["response_request_id"]], key=lambda e: e["generated_output_tokens_total"])
                assert [e["generated_output_tokens_total"] for e in produced] == list(range(1, len(arrivals) + 1)), row["source_request_id"]
                if len(arrivals) > 1:
                    intervals = [b - a for a, b in zip(arrivals, arrivals[1:])]
                    assert all(x >= 0 for x in intervals), row["source_request_id"]
                    per_request_tpot.append(statistics.mean(intervals) * 1000)
                    gaps.extend(x * 1000 for x in intervals)
                    production_intervals = [b["monotonic"] - a["monotonic"] for a, b in zip(produced, produced[1:])]
                    assert all(x >= 0 for x in production_intervals), row["source_request_id"]
                    stage0_tpot.append(statistics.mean(production_intervals) * 1000)
                    stage0_gaps.extend(x * 1000 for x in production_intervals)
            report["arms"][arm][tier] = {
                "requests": len(rows),
                "ttft_s": stats([r["ttft_s"] for r in rows if r["ttft_s"] is not None]),
                "ttft_slo_met": sum(r["first_slo_met"] for r in score_rows),
                "visible_tpot_ms": stats(per_request_tpot),
                "pooled_gap_ms": stats(gaps),
                "stage0_tpot_ms": stats(stage0_tpot),
                "stage0_pooled_gap_ms": stats(stage0_gaps),
                "reading_rebuffer_s": stats([r["rebuffer_s"] for r in score_rows]),
                "reading_max_rebuffer_s": stats([r["max_rebuffer_s"] for r in score_rows]),
                "peak_client_unread_s": stats([r["peak_client_buffer_s"] for r in score_rows]),
            }
    output = root / "latency_breakdown.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    for root in args.roots:
        data = summarize(root)
        print(root)
        print("arm tier n TTFT_p50_s TTFT_p95_s TPOT_p50_ms TPOT_p95_ms gap_p95_ms gap_max_ms stage0_TPOT_p50_ms stage0_TPOT_p95_ms rebuffer_max_s first_SLO")
        for arm, groups in data["arms"].items():
            for tier, v in groups.items():
                t, p, g, s = v["ttft_s"], v["visible_tpot_ms"], v["pooled_gap_ms"], v["stage0_tpot_ms"]
                print(f"{arm} {tier} {v['requests']} {t['p50']:.3f} {t['p95']:.3f} "
                      f"{p['p50']:.1f} {p['p95']:.1f} {g['p95']:.1f} {g['max']:.1f} "
                      f"{s['p50']:.1f} {s['p95']:.1f} {v['reading_max_rebuffer_s']['max']:.3f} {v['ttft_slo_met']}/{v['requests']}")


if __name__ == "__main__":
    main()
