"""Replay a frozen ServeGen workload without changing the historical run ledger.

The tested sender and file:// media conversion are reused from the archived
ServeGen study. This runner keeps source request IDs stable across runs.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from decimal import Decimal
import hashlib
import json
import statistics
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ARCHIVED_CLIENT = Path("/home/zhongyu/research_reports/servegen_active_buffer_20261012")
sys.path.insert(0, str(ARCHIVED_CLIENT))
from replay import materialize  # noqa: E402
from sse_client_extended import atomic_json, send_request  # noqa: E402


def get_json(endpoint: str, path: str) -> dict:
    with urllib.request.urlopen(endpoint.rstrip("/") + path, timeout=8) as response:
        return json.load(response)


def audio_padding_entries(workload: Path) -> dict[str, dict]:
    """Reuse the archived CPU/server audio-padding audit for an arrival-only trace."""
    workload_sha256 = hashlib.sha256(workload.read_bytes()).hexdigest()
    source_sha256 = workload_sha256
    manifest_path = workload.with_suffix(".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["output_sha256"] != workload_sha256:
            raise ValueError("workload bytes differ from its frozen manifest")
        kind = manifest.get("kind")
        if kind == "arrival_scaled_from_existing_multimodal_trace":
            if manifest.get("request_ids_and_all_nonarrival_fields_unchanged") is not True:
                raise ValueError("arrival-only workload differs from its source")
        elif kind == "request_replicated_and_arrival_scaled":
            source = Path(manifest["source"])
            if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source_sha256"]:
                raise ValueError("replicated workload source hash differs from manifest")
            originals = {str(row["source_request_id"]): row for row in
                         (json.loads(line) for line in source.read_text().splitlines() if line.strip())}
            replicas = [json.loads(line) for line in workload.read_text().splitlines() if line.strip()]
            factor = int(manifest["request_factor"])
            if len(replicas) != len(originals) * factor or manifest.get("contents_are_repeated") is not True:
                raise ValueError("replicated workload count or provenance invalid")
            seen = Counter()
            for row in replicas:
                original_id = str(row["replica_of_source_request_id"])
                original = originals.get(original_id)
                replica_index = row["replica_index"]
                if original is None or not isinstance(replica_index, int) or not 0 <= replica_index < factor:
                    raise ValueError("invalid replica provenance")
                seen[(original_id, replica_index)] += 1
                suffix = f"__replica{replica_index}"
                if (row["source_request_id"] != original_id + suffix or
                        row["request_id"] != str(original["request_id"]) + suffix or
                        row["arrival_offset_s"] != float(Decimal(str(original["arrival_offset_s"])) /
                                                          Decimal(str(manifest["time_factor"])))):
                    raise ValueError("replica ID or arrival differs from frozen rule")
                for key, value in original.items():
                    if key not in ("source_request_id", "request_id", "arrival_offset_s") and row.get(key) != value:
                        raise ValueError(f"replica field {key} differs from source")
            if len(seen) != len(replicas) or any(count != 1 for count in seen.values()):
                raise ValueError("replica ID is missing or repeated")
        else:
            raise ValueError(f"unsupported workload manifest kind: {kind}")
        source_sha256 = manifest["source_sha256"]
    audit_path = ARCHIVED_CLIENT / "audio_padding_validation.json"
    if not audit_path.exists():
        return {}
    entries = json.loads(audit_path.read_text())["entries"]
    return {entry["request_id"]: entry for entry in entries
            if entry["workload_sha256"] == source_sha256}


def trace_offsets(directory: Path) -> dict[str, int]:
    if not directory.is_dir():
        return {}
    return {str(path.relative_to(directory)): path.stat().st_size
            for path in directory.rglob("*.jsonl")}


def copy_trace_slice(directory: Path, output: Path, before: dict[str, int]) -> None:
    for name, end in trace_offsets(directory).items():
        destination = output / "traces" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (directory / name).open("rb") as source, destination.open("xb") as target:
            start = before.get(name, 0)
            source.seek(start)
            remaining = end - start
            while remaining > 0:
                chunk = source.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise IOError(f"trace ended before recorded offset: {name}")
                target.write(chunk)
                remaining -= len(chunk)


def wait_drained(endpoint: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        try:
            last = get_json(endpoint, "/v1/omni/drain_status?timeout=2")
            if last.get("healthy") and last.get("drained"):
                return last
        except Exception as exc:
            last = {"error": f"{type(exc).__name__}: {exc}"}
        time.sleep(0.5)
    raise TimeoutError(f"service did not drain within {timeout_s}s: {last}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8095")
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--timeout-s", type=float, default=240)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.workload.read_text().splitlines() if line.strip()]
    ids = [str(row["source_request_id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        parser.error("workload must have unique source_request_id values")
    if any(not 0 <= row["arrival_offset_s"] < args.duration_s for row in rows):
        parser.error("every arrival must be inside duration-s")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run: {args.output_dir}")
    service = json.loads((args.session_dir / "service.json").read_text())
    framework = Path(__file__).resolve().parents[2]
    if Path(service["framework"]).resolve() != framework:
        raise ValueError("service framework is not this fixed experiment worktree")
    port = urllib.parse.urlsplit(args.endpoint).port
    if service["port"] != port:
        raise ValueError("endpoint port differs from the recorded session")
    model = get_json(args.endpoint, "/v1/models")["data"][0]["id"]
    drained_before = wait_drained(args.endpoint, 30)
    audio_expected = audio_padding_entries(args.workload)
    adjusted_audio_request_ids = []
    prepared = []
    for row in rows:
        record = copy.deepcopy(row)
        cpu = audio_expected.get(record.get("replica_of_source_request_id",
                                        record["source_request_id"]))
        if cpu is not None:
            adjusted_audio_request_ids.append(record["source_request_id"])
            if record["actual_processor_input_tokens"] != cpu["original_expected_tokens"]:
                raise ValueError("audio prompt count differs from archived audit")
            record["initial_hf_unpadded_expected_tokens"] = record["actual_processor_input_tokens"]
            record["actual_processor_input_tokens"] = cpu["cpu_padded_expected_tokens"]
            record["server_aligned_prompt_ids_json_sha256"] = cpu["cpu_prompt_ids_json_sha256"]
            record["offline_audio_preprocessing"] = (
                "right-zero-pad to feature extractor hop length, matching server"
            )
        record["payload"] = materialize(record["payload"])
        record["payload"]["model"] = model
        prepared.append(record)

    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir / "protocol.json", {
        "schema_version": 1,
        "kind": "open_loop_real_multimodal_replay",
        "workload": str(args.workload.resolve()),
        "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        "request_count": len(rows),
        "arrival_window_s": args.duration_s,
        "timeout_s": args.timeout_s,
        "timeout_is_safety_limit_not_slo": True,
        "model": model,
        "service": service,
        "drained_before": drained_before,
        "client_source": str(ARCHIVED_CLIENT),
        "audio_padding_audit": str(ARCHIVED_CLIENT / "audio_padding_validation.json"),
        "audio_padding_adjusted_requests": sorted(adjusted_audio_request_ids),
    })
    traces = args.session_dir / "traces"
    before = trace_offsets(traces)
    origin = time.monotonic() + 1.0
    results = []
    run_error = None
    drained_after = None

    def send(record: dict) -> dict:
        due = origin + record["arrival_offset_s"]
        time.sleep(max(0.0, due - time.monotonic()))
        return send_request(
            record, args.output_dir / "client", args.endpoint + "/v1/chat/completions",
            scheduled_arrival_monotonic_s=due, timeout_s=args.timeout_s,
        )

    try:
        with ThreadPoolExecutor(max_workers=len(prepared)) as pool:
            futures = [pool.submit(send, row) for row in prepared]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps({"completed": len(results), "of": len(rows),
                                  "status": result["status"],
                                  "request_id": result["source_request_id"]}), flush=True)
        time.sleep(max(0.0, origin + args.duration_s - time.monotonic()))
        drained_after = wait_drained(args.endpoint, 90)
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
    finally:
        copy_trace_slice(traces, args.output_dir, before)
        lateness = [float(row["send_lateness_s"]) for row in results
                    if row.get("send_lateness_s") is not None]
        atomic_json(args.output_dir / "results.json", {
            "schema_version": 1,
            "planned_requests": len(rows),
            "attempted_requests": len(results),
            "success": sum(row["status"] == "success" for row in results),
            "error": sum(row["status"] == "error" for row in results),
            "timeout": sum(row["status"] == "timeout" for row in results),
            "origin_monotonic_s": origin,
            "arrival_window_s": args.duration_s,
            "send_lateness_p50_s": statistics.median(lateness) if lateness else None,
            "send_lateness_p95_s": (sorted(lateness)[min(len(lateness) - 1, int(0.95 * len(lateness)))]) if lateness else None,
            "send_lateness_max_s": max(lateness) if lateness else None,
            "drained_after": drained_after,
            "run_error": run_error,
            "results": results,
        })
    if run_error or len(results) != len(rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
