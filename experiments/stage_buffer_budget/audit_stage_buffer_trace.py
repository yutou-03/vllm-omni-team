#!/usr/bin/env python3
"""Audit receiver-ready inventory and hold evidence from a stage trace.

Chunk identity includes the pipeline/stage attempt. An abandoned attempt may
have published chunks which the next attempt must never count as ready.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


EVENTS = {
    "stage_study_text_payload_sent", "stage_study_text_payload_received",
    "stage_study_text_conditioning_selected", "stage_study_chunk_sent",
    "stage_study_chunk_received", "stage_study_chunk_conversion_started",
}


def units(mapping):
    return sum(value for value in mapping.values() if isinstance(value, (int, float)))


def fresh():
    return dict(text_sent={}, text_received={}, prompt=set(), selected=set(),
                codec_sent={}, codec_received={}, codec_started={}, errors=[])


def audit(path: Path) -> dict:
    attempts = defaultdict(fresh)
    event_counts = Counter()
    decisions = defaultdict(lambda: dict(reasons=Counter(), peak=None,
                                         receiver_ready_max=0, published_max=0,
                                         min_high_s=None, max_guard_s=0,
                                         victim_rejections=Counter(), rejection_peak=None))
    hold_events = Counter()
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for line in source:
            digest.update(line)
            event = json.loads(line)
            name = event.get("event")
            event_counts[name] += 1
            if name == "stage_buffer_decision":
                stage = int(event["stage_id"])
                row = decisions[stage]
                row["reasons"][event["reason"]] += 1
                row["victim_rejections"].update(event.get("rejection_counts") or {})
                example = event.get("rejection_example")
                if example is not None:
                    coverage = example.get("coverage_s")
                    prior = row["rejection_peak"]
                    if prior is None or (coverage is not None and
                                         (prior.get("coverage_s") is None or coverage > prior["coverage_s"])):
                        row["rejection_peak"] = {key: example.get(key) for key in (
                            "victim_request_id", "rejection_reason", "decode_boundary",
                            "inventory_units", "receiver_ready_inventory_units",
                            "coverage_s", "high_s", "requested_hold_s", "num_output_tokens")}
                for key, source_key in (("receiver_ready_max", "receiver_ready_inventory_units"),
                                        ("published_max", "published_inventory_units")):
                    value = event.get(source_key)
                    if value is not None:
                        row[key] = max(row[key], value)
                high = event.get("high_s")
                if high is not None:
                    row["min_high_s"] = high if row["min_high_s"] is None else min(row["min_high_s"], high)
                row["max_guard_s"] = max(row["max_guard_s"], event.get("resume_guard_s") or 0)
                coverage = event.get("coverage_s")
                if coverage is not None and (row["peak"] is None or coverage > row["peak"]["coverage_s"]):
                    row["peak"] = {key: event.get(key) for key in (
                        "logical_request_id", "decision_monotonic_s", "coverage_s",
                        "inventory_units", "receiver_ready_inventory_units",
                        "published_inventory_units", "consumption_rate_units_s",
                        "high_s", "low_s", "resume_guard_s", "requested_hold_s")}
            if name.startswith("small_hold_") or name.startswith("urgent_keep_"):
                hold_events[name] += 1
            if name not in EVENTS:
                continue
            key = (str(event.get("logical_request_id") or event.get("request_id")),
                   event.get("pipeline_attempt_id", 0), event.get("stage_attempt_id", 0))
            state = attempts[key]
            if name == "stage_study_text_conditioning_selected":
                if event.get("conditioning_kind") == "output_embedding":
                    state["selected"].update(range(event["start_index"], event["end_index"]))
                continue
            item = {"stage_study_text_payload_sent": "text_sent",
                    "stage_study_text_payload_received": "text_received",
                    "stage_study_chunk_sent": "codec_sent",
                    "stage_study_chunk_received": "codec_received",
                    "stage_study_chunk_conversion_started": "codec_started"}[name]
            value = (event.get("raw_entity_embedding_rows_total") if item.startswith("text")
                     else event.get("new_frames"))
            chunk = event["chunk_id"]
            old = state[item].setdefault(chunk, value)
            if old != value or not isinstance(value, (int, float)) or value < 0:
                state["errors"].append(f"invalid or conflicting {item} chunk {chunk}")
            if item.startswith("text"):
                if event.get("raw_entity_embedding_layout_valid") is not True:
                    state["errors"].append(f"invalid embedding layout at {item} chunk {chunk}")
                if event.get("raw_ids_prompt_count") is not None:
                    state["prompt"].add(event["raw_ids_prompt_count"])
    violations = []
    unreceived = []
    totals = Counter()
    for key, state in attempts.items():
        errors = state["errors"][:]
        if len(state["prompt"]) > 1:
            errors.append("prompt count changed")
        for downstream, upstream in (("text_received", "text_sent"),
                                     ("codec_received", "codec_sent"),
                                     ("codec_started", "codec_received")):
            for chunk, value in state[downstream].items():
                if state[upstream].get(chunk) != value:
                    errors.append(f"{downstream} chunk {chunk} absent or different upstream")
        for item in ("text_sent", "text_received", "codec_sent", "codec_received", "codec_started"):
            totals[item + "_chunks"] += len(state[item])
            totals[item + "_units"] += units(state[item])
        if state["text_received"] and state["prompt"]:
            ready = units(state["text_received"]) - next(iter(state["prompt"]))
            if len(state["selected"]) > ready:
                errors.append("selected positions exceed received output rows")
        if units(state["codec_started"]) > units(state["codec_received"]):
            errors.append("started codec frames exceed received frames")
        missing = sorted(set(state["text_sent"]) - set(state["text_received"]))
        if missing:
            unreceived.append(dict(request_id=key[0], pipeline_attempt_id=key[1],
                                   stage_attempt_id=key[2], chunks=len(missing),
                                   units=sum(state["text_sent"][i] for i in missing if isinstance(state["text_sent"][i], (int, float))),
                                   first_chunk=missing[0], last_chunk=missing[-1]))
        if errors:
            violations.append(dict(request_id=key[0], pipeline_attempt_id=key[1],
                                   stage_attempt_id=key[2], errors=errors))
    return dict(trace=str(path.resolve()), trace_sha256=digest.hexdigest(),
                attempt_count=len(attempts), event_counts={k: v for k, v in event_counts.items()
                                                  if k in EVENTS or "hold" in k or "urgent_keep" in k},
                conservation_totals=dict(totals), conservation_violations=violations,
                published_not_received=unreceived,
                stages={str(k): {**v, "reasons": dict(v["reasons"]),
                                  "victim_rejections": dict(v["victim_rejections"])} for k, v in decisions.items()},
                hold_events=dict(hold_events))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.trace)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    if report["conservation_violations"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
