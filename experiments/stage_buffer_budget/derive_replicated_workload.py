#!/usr/bin/env python3
"""Replicate a frozen trace and compress its planned arrivals, without new content."""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-factor", type=int, required=True)
    parser.add_argument("--time-factor", type=Decimal, required=True)
    parser.add_argument("--source-window-s", type=Decimal, required=True)
    args = parser.parse_args()
    if args.request_factor < 2 or args.time_factor <= 1 or args.source_window_s <= 0:
        parser.error("request-factor must be >=2; time-factor >1; source-window-s >0")
    if args.source.resolve() == args.output.resolve():
        parser.error("source and output must differ")
    source_bytes = args.source.read_bytes()
    source_rows = [json.loads(line) for line in source_bytes.splitlines() if line.strip()]
    if not source_rows:
        parser.error("source contains no requests")
    ids = [str(row["source_request_id"]) for row in source_rows]
    if len(ids) != len(set(ids)):
        parser.error("source request IDs are not unique")
    previous = -math.inf
    derived_rows = []
    for original in source_rows:
        arrival = float(original["arrival_offset_s"])
        if not math.isfinite(arrival) or arrival < previous or arrival < 0:
            parser.error("source arrivals must be finite, nonnegative, and ordered")
        if Decimal(str(arrival)) >= args.source_window_s:
            parser.error("source arrival lies outside source-window-s")
        previous = arrival
        for replica_index in range(args.request_factor):
            row = dict(original)
            row["arrival_offset_s"] = float(Decimal(str(arrival)) / args.time_factor)
            row["replica_of_source_request_id"] = str(original["source_request_id"])
            row["replica_index"] = replica_index
            suffix = f"__replica{replica_index}"
            row["source_request_id"] = str(original["source_request_id"]) + suffix
            row["request_id"] = str(original["request_id"]) + suffix
            derived_rows.append(row)
    if len({row["source_request_id"] for row in derived_rows}) != len(derived_rows):
        parser.error("derived request IDs are not unique")
    output_bytes = ("\n".join(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                           separators=(",", ":")) for row in derived_rows)
                    + "\n").encode()
    manifest = {
        "schema_version": 1,
        "kind": "request_replicated_and_arrival_scaled",
        "source": str(args.source.resolve()),
        "source_sha256": sha256(source_bytes),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(output_bytes),
        "transformation": "for each source request create request_factor unique IDs at source arrival/time_factor",
        "request_factor": args.request_factor,
        "time_factor": str(args.time_factor),
        "source_window_s": float(args.source_window_s),
        "derived_window_s": float(args.source_window_s / args.time_factor),
        "source_request_count": len(source_rows),
        "request_count": len(derived_rows),
        "source_rate_rps": float(Decimal(len(source_rows)) / args.source_window_s),
        "derived_rate_rps": float(Decimal(len(derived_rows)) * args.time_factor / args.source_window_s),
        "relative_rate_factor": float(args.request_factor * args.time_factor),
        "source_last_arrival_s": float(source_rows[-1]["arrival_offset_s"]),
        "derived_last_arrival_s": derived_rows[-1]["arrival_offset_s"],
        "source_modality_counts": dict(Counter(row.get("input_modality", "unknown") for row in source_rows)),
        "derived_modality_counts": dict(Counter(row.get("input_modality", "unknown") for row in derived_rows)),
        "distinct_source_payloads": len(source_rows),
        "contents_are_repeated": True,
        "replicas_share_planned_arrival": True,
        "identity_fields_changed": ["source_request_id", "request_id"],
        "all_other_nonarrival_fields_unchanged": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, contents in ((args.output, output_bytes),
                           (args.output.with_suffix(".manifest.json"),
                            (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())):
        if path.exists():
            if path.read_bytes() != contents:
                raise FileExistsError(f"refusing to overwrite different file: {path}")
        else:
            path.write_bytes(contents)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
