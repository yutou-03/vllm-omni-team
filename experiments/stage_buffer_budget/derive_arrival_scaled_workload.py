"""Make a denser arrival trace without changing requests or their order."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=Decimal, default=Decimal("1.2"))
    parser.add_argument("--source-window-s", type=Decimal, default=Decimal("120"))
    args = parser.parse_args()

    if not Decimal("1") < args.factor <= Decimal("4"):
        parser.error("factor must be greater than 1 and at most 4")
    if args.source_window_s <= 0:
        parser.error("source-window-s must be positive")
    if args.source.resolve() == args.output.resolve():
        parser.error("source and output must differ")

    source_bytes = args.source.read_bytes()
    source_rows = [json.loads(line) for line in source_bytes.splitlines() if line.strip()]
    if not source_rows:
        parser.error("source contains no requests")
    ids = [str(row["request_id"]) for row in source_rows]
    if len(ids) != len(set(ids)):
        parser.error("source request IDs are not unique")

    derived_rows = []
    previous = -math.inf
    for row in source_rows:
        arrival = float(row["arrival_offset_s"])
        if not math.isfinite(arrival) or arrival < previous or arrival < 0:
            parser.error("source arrival offsets must be finite and ordered")
        if Decimal(str(arrival)) >= args.source_window_s:
            parser.error("source arrival lies outside the source window")
        previous = arrival
        derived = dict(row)
        derived["arrival_offset_s"] = float(Decimal(str(arrival)) / args.factor)
        derived_rows.append(derived)

    output_bytes = ("\n".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        for row in derived_rows
    ) + "\n").encode()
    manifest = {
        "schema_version": 1,
        "kind": "arrival_scaled_from_existing_multimodal_trace",
        "source": str(args.source.resolve()),
        "source_sha256": sha256(source_bytes),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(output_bytes),
        "transformation": "arrival_offset_s = source_arrival_offset_s / factor",
        "factor": str(args.factor),
        "request_count": len(derived_rows),
        "source_window_s": float(args.source_window_s),
        "derived_window_s": float(args.source_window_s / args.factor),
        "source_rate_rps": float(Decimal(len(derived_rows)) / args.source_window_s),
        "derived_rate_rps": float(Decimal(len(derived_rows)) * args.factor / args.source_window_s),
        "source_last_arrival_s": float(source_rows[-1]["arrival_offset_s"]),
        "derived_last_arrival_s": derived_rows[-1]["arrival_offset_s"],
        "modality_counts": dict(Counter(row.get("input_modality", "unknown") for row in source_rows)),
        "request_ids_and_all_nonarrival_fields_unchanged": all(
            {k: v for k, v in original.items() if k != "arrival_offset_s"}
            == {k: v for k, v in derived.items() if k != "arrival_offset_s"}
            for original, derived in zip(source_rows, derived_rows)
        ),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, content in (
        (args.output, output_bytes),
        (manifest_path, (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()),
    ):
        if path.exists():
            if path.read_bytes() != content:
                raise FileExistsError(f"refusing to overwrite different file: {path}")
        else:
            path.write_bytes(content)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
