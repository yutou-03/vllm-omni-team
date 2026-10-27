#!/usr/bin/env python3
"""Bounded, open-loop SSE client for extended ratio replay.

Vendored from servegen_active_buffer_20261012/sse_client.py; only the
per-request safety ceiling changed from 240 to 900 seconds. Original SHA-256:
de349a4df03f2e5f2c676dfed756778fc6d65c811100a203831996ca25f66d63

Original description:

Every received SSE byte and data event is retained. Audio feedback measures
decoded samples at the client, never codec tokens or base64 size. Playback
starts on the first valid audio chunk and resumes immediately after starvation.
All clocks are time.monotonic() on the shared host.
"""
from __future__ import annotations

import argparse
import base64
import bisect
import hashlib
import http.client
import io
import json
import math
import socket
import threading
import time
import urllib.parse
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class Playback:
    """Work-conserving, 1x playback; startup wait is reported separately."""

    def __init__(self):
        self.first = None
        self.end = None
        self.duration = 0.0
        self.stalls = []

    def receive(self, arrival, duration):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("audio duration must be finite and positive")
        if self.first is None:
            self.first = arrival
            self.end = arrival
        if arrival > self.end:
            self.stalls.append({"start_monotonic_s": self.end,
                                "end_monotonic_s": arrival,
                                "duration_s": arrival - self.end})
        self.end = max(arrival, self.end) + duration
        self.duration += duration

    def feedback(self, now):
        return {"first_audio_monotonic_s": self.first,
                "cumulative_received_playable_s": self.duration,
                "playback_end_monotonic_s": self.end,
                "projected_buffered_s": max(0.0, self.end - now) if self.end else 0.0,
                "projection_monotonic_s": now,
                "observed_stall_s": sum(x["duration_s"] for x in self.stalls)}


class FeedbackWriter:
    """One writer object per replay. File replacement is atomic for readers.

    A reader must age/project from playback_end_monotonic_s; the stored buffer
    is a snapshot, not a continuously refreshed assertion of current inventory.
    """

    def __init__(self, path):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        self.requests = {}
        if self.path:
            atomic_json(self.path, {"schema_version": 1, "requests": {}})

    def update(self, request_id, **fields):
        if not self.path:
            return
        with self.lock:
            self.requests.setdefault(request_id, {}).update(fields)
            now = time.monotonic()
            atomic_json(self.path, {"schema_version": 1,
                                   "written_monotonic_s": now,
                                   "requests": self.requests})


def decode_audio(raw):
    """Decode each complete WAV, rejecting short payloads and zero samples."""
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav:
            frames, rate, channels, width = (wav.getnframes(), wav.getframerate(),
                                            wav.getnchannels(), wav.getsampwidth())
            pcm = wav.readframes(frames)
            if len(pcm) != frames * channels * width:
                raise ValueError("truncated PCM WAV payload")
            if wav.getcomptype() != "NONE":
                raise ValueError("unsupported compressed WAV")
            decoded_hash = sha256(pcm)
            decoded_repr = f"PCM_{width * 8} native WAV bytes"
    except wave.Error:
        import soundfile as sf
        with sf.SoundFile(io.BytesIO(raw)) as audio:
            frames, rate, channels = len(audio), audio.samplerate, audio.channels
            samples = audio.read(dtype="float64", always_2d=True)
            if len(samples) != frames:
                raise ValueError("truncated encoded audio payload")
            decoded_hash = sha256(samples.astype("<f8").tobytes())
            decoded_repr = "float64 little-endian interleaved samples"
    if rate <= 0 or frames <= 0 or channels <= 0:
        raise ValueError("audio has no playable samples")
    return {"frames": frames, "sample_rate": rate, "channels": channels,
            "duration_s": frames / rate, "encoded_sha256": sha256(raw),
            "decoded_sha256": decoded_hash, "decoded_representation": decoded_repr}


class DeadlineWatchdog:
    """Interrupt an outstanding socket read, including a slow-drip response."""

    def __init__(self, timeout_s):
        self.expired = threading.Event()
        self.lock = threading.Lock()
        self.sock = None
        self.timer = threading.Timer(timeout_s, self._expire)
        self.timer.daemon = True

    def _expire(self):
        self.expired.set()
        with self.lock:
            if self.sock:
                try:
                    self.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def attach(self, sock):
        with self.lock:
            self.sock = sock
        if self.expired.is_set():
            self._expire()
            raise TimeoutError("request wall deadline elapsed during connect")

    def check(self):
        if self.expired.is_set():
            raise TimeoutError("request wall deadline elapsed")

    def cancel(self):
        self.timer.cancel()


def iter_sse(response, raw_file, watchdog):
    data_lines = []
    last_arrival = None
    while True:
        watchdog.check()
        raw = response.readline()
        arrival = time.monotonic()
        if raw:
            raw_file.write(raw)
        watchdog.check()
        if not raw:
            if data_lines:
                # Preserve a final event even when the server omitted its blank line.
                yield last_arrival, "\n".join(data_lines)
            return
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                yield arrival, "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field == "data":
            data_lines.append(value[1:] if separator and value.startswith(" ") else value)
            last_arrival = arrival


class Capture:
    def __init__(self, request_id, directory, feedback):
        self.request_id, self.directory, self.feedback = request_id, directory, feedback
        self.playback = Playback()
        self.text_parts, self.text_ends, self.text_times = [], [], []
        self.token_ids, self.token_times = [], []
        self.audio_token_ids = []
        self.audio_chunks = []
        self.response_id = None
        self.usage = None
        self.text_prompt_token_ids = None
        self.first_text = self.text_finish = self.last_audio = None
        self.finish_events = []
        self.done = False
        self.event_count = 0
        self.decode_elapsed_s = 0.0

    def event(self, arrival, data):
        self.event_count += 1
        if data == "[DONE]":
            self.done = True
            return
        chunk = json.loads(data)
        if not isinstance(chunk, dict):
            raise ValueError("SSE chunk is not an object")
        if chunk.get("error") is not None:
            raise ValueError(f"SSE error: {chunk['error']}")
        response_id = chunk.get("id")
        if response_id is not None:
            if not isinstance(response_id, str) or not response_id:
                raise ValueError("invalid response request ID")
            if self.response_id is not None and response_id != self.response_id:
                raise ValueError("response request ID changed")
            self.response_id = response_id
        if chunk.get("usage") is not None:
            self.usage = chunk["usage"]
        modality = chunk.get("modality")
        if modality not in (None, "text", "audio"):
            raise ValueError(f"unsupported response modality {modality!r}")
        if modality != "audio" and chunk.get("prompt_token_ids") is not None:
            prompt_ids = chunk["prompt_token_ids"]
            if not isinstance(prompt_ids, list) or any(type(t) is not int for t in prompt_ids):
                raise ValueError("invalid text prompt token IDs")
            if self.text_prompt_token_ids is not None and prompt_ids != self.text_prompt_token_ids:
                raise ValueError("text prompt token IDs changed")
            self.text_prompt_token_ids = prompt_ids
        for choice in chunk.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("pilot supports exactly one output choice")
            delta = choice.get("delta") or {}
            content = delta.get("content")
            audio = content if modality == "audio" else None
            if modality != "audio" and isinstance(content, str) and content:
                self.text_parts.append(content)
                self.text_ends.append((self.text_ends[-1] if self.text_ends else 0) + len(content))
                self.text_times.append(arrival)
                if self.first_text is None:
                    self.first_text = arrival
            if delta.get("audio"):
                if audio:
                    raise ValueError("ambiguous duplicate audio payload fields")
                audio = delta["audio"].get("data")
            tokens = choice.get("token_ids")
            if tokens is not None:
                if not isinstance(tokens, list) or any(type(t) is not int for t in tokens):
                    raise ValueError("invalid output token IDs")
                if modality == "audio":
                    self.audio_token_ids.extend(tokens)
                else:
                    self.token_ids.extend(tokens)
                    self.token_times.extend([arrival] * len(tokens))
            if choice.get("finish_reason") is not None:
                self.finish_events.append({"arrival_monotonic_s": arrival,
                                           "modality": modality,
                                           "finish_reason": choice["finish_reason"],
                                           "stop_reason": choice.get("stop_reason")})
                # Each Qwen audio chunk carries stop, so only DONE finishes audio.
                if modality != "audio":
                    self.text_finish = arrival
            if audio:
                start = time.monotonic()
                raw = base64.b64decode(audio, validate=True)
                info = decode_audio(raw)
                self.decode_elapsed_s += time.monotonic() - start
                sequence = len(self.audio_chunks)
                path = self.directory / f"audio_{sequence:04d}.wav"
                path.write_bytes(raw)
                info.update({"sequence": sequence, "path": path.name,
                             "arrival_monotonic_s": arrival,
                             "position_s": self.playback.duration})
                self.audio_chunks.append(info)
                self.playback.receive(arrival, info["duration_s"])
                self.last_audio = arrival
            now = time.monotonic()
            self.feedback.update(self.request_id, response_request_id=self.response_id,
                                 last_receive_monotonic_s=arrival,
                                 last_audio_receive_monotonic_s=self.last_audio,
                                 received_text_token_count=len(self.token_ids),
                                 audio_chunk_count=len(self.audio_chunks),
                                 **self.playback.feedback(now))


def text_pacing(times, origin, first_slo_s=2.0, rate=20.0):
    late = [max(0.0, t - (origin + first_slo_s + i / rate)) for i, t in enumerate(times)]
    return {"units": len(times), "on_time_fraction": sum(x <= 1e-9 for x in late) / len(late) if late else None,
            "max_lateness_s": max(late, default=0.0), "rate_units_per_s": rate}


def _validate_complete(capture, payload):
    if not capture.done:
        raise ValueError("stream ended before [DONE]")
    if capture.response_id is None or capture.first_text is None:
        raise ValueError("complete stream requires response ID and visible text")
    expects_audio = "audio" in payload.get("modalities", [])
    if expects_audio != bool(capture.audio_chunks):
        raise ValueError("audio output does not match requested modalities")
    if payload.get("return_token_ids") and not capture.token_ids:
        raise ValueError("requested text token IDs are absent")
    if payload.get("ignore_eos") and payload.get("max_tokens") is not None and len(capture.token_ids) != payload["max_tokens"]:
        raise ValueError(f"fixed output budget mismatch: expected {payload['max_tokens']}, got {len(capture.token_ids)}")
    if payload.get("stream_options", {}).get("include_usage"):
        if not isinstance(capture.usage, dict):
            raise ValueError("requested final usage is absent")
        values = [capture.usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")]
        if any(type(x) is not int or x < 0 for x in values) or values[0] + values[1] != values[2]:
            raise ValueError("invalid final usage accounting")
        if payload.get("return_token_ids") and values[1] != len(capture.token_ids):
            raise ValueError("final completion_tokens differs from received text token IDs")


def send_request(record, output_dir, endpoint, feedback=None,
                 scheduled_arrival_monotonic_s=None, timeout_s=120.0):
    """Send one frozen record and retain a result even on timeout/protocol error."""
    if not 0 < timeout_s <= 900:
        raise ValueError("ratio replay safety deadline must be in (0, 900] seconds")
    request_id = record["source_request_id"]
    payload = record["payload"]
    directory = Path(output_dir) / ("request_" + sha256(request_id.encode())[:16])
    directory.mkdir(parents=True, exist_ok=False)
    feedback = feedback or FeedbackWriter(None)
    capture = Capture(request_id, directory, feedback)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    start = time.monotonic()
    scheduled = start if scheduled_arrival_monotonic_s is None else scheduled_arrival_monotonic_s
    deadline = scheduled + timeout_s
    remaining = max(0.0, deadline - start)
    connection = connection_class(parsed.hostname, parsed.port, timeout=max(.001, remaining))
    watchdog = DeadlineWatchdog(remaining)
    status, error, http_status = "success", None, None
    response = None
    watchdog.timer.start()
    feedback.update(request_id, status="in_flight", send_monotonic_s=start,
                    scheduled_arrival_monotonic_s=scheduled,
                    deadline_monotonic_s=deadline,
                    modalities=payload.get("modalities", []),
                    **capture.playback.feedback(start))
    try:
        watchdog.check()
        connection.connect()
        watchdog.attach(connection.sock)
        connection.request("POST", target, body=encoded,
                           headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                                    "x-request-id": request_id})
        response = connection.getresponse()
        http_status = response.status
        atomic_json(directory / "response_headers.json", dict(response.getheaders()))
        with (directory / "raw.sse").open("wb") as raw_file, (directory / "events.jsonl").open("w") as event_file:
            if not 200 <= http_status < 300:
                raw_file.write(response.read())
                raise ValueError(f"HTTP status {http_status}")
            for arrival, data in iter_sse(response, raw_file, watchdog):
                event_file.write(json.dumps({"arrival_monotonic_s": arrival, "data": data}, ensure_ascii=False) + "\n")
                event_file.flush()
                capture.event(arrival, data)
                watchdog.check()
                if capture.done:
                    break
        _validate_complete(capture, payload)
        expected_prompt = record.get("actual_processor_input_tokens")
        actual_prompt = len(capture.text_prompt_token_ids) if capture.text_prompt_token_ids is not None else None
        if expected_prompt is not None and actual_prompt != expected_prompt:
            raise ValueError(f"invalid workload: expected {expected_prompt} prompt tokens, "
                             f"first text SSE contained {actual_prompt}")
    except Exception as exc:
        status = "timeout" if watchdog.expired.is_set() or isinstance(exc, (TimeoutError, socket.timeout)) else "error"
        error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    finally:
        completion = time.monotonic()
        watchdog.cancel()
        if response is not None:
            response.close()
        connection.close()
    text = "".join(capture.text_parts)
    (directory / "text.txt").write_text(text, encoding="utf-8")
    stall = sum(x["duration_s"] for x in capture.playback.stalls)
    audio_s = capture.playback.duration
    result = {"source_request_id": request_id, "status": status, "error": error,
              "http_status": http_status, "response_request_id": capture.response_id,
              "request_payload_sha256": sha256(encoded), "artifact_directory": str(directory),
              "scheduled_arrival_monotonic_s": scheduled, "send_monotonic_s": start,
              "send_lateness_s": start - scheduled, "deadline_monotonic_s": deadline,
              "client_completion_monotonic_s": completion, "e2e_s": completion - scheduled,
              "e2e_from_send_s": completion - start,
              "e2e_from_scheduled_s": completion - scheduled,
              "first_text_monotonic_s": capture.first_text,
              "text_finish_monotonic_s": capture.text_finish,
              "first_audio_monotonic_s": capture.playback.first,
              "last_audio_monotonic_s": capture.last_audio,
              "ttft_s": capture.first_text - scheduled if capture.first_text is not None else None,
              "ttfp_s": capture.playback.first - scheduled if capture.playback.first is not None else None,
              "done_received": capture.done, "event_count": capture.event_count,
              "finish_events": capture.finish_events, "usage": capture.usage,
              "text_sha256": sha256(text.encode()), "text_characters": len(text),
              "text_token_ids": capture.token_ids,
              "text_token_ids_sha256": sha256(json.dumps(capture.token_ids, separators=(",", ":")).encode()),
              "text_token_arrival_monotonic_s": capture.token_times,
              "text_character_ends": capture.text_ends, "text_chunk_arrival_monotonic_s": capture.text_times,
              "audio_token_ids": capture.audio_token_ids,
              "generated_token_pacing": text_pacing(capture.token_times, scheduled),
              "audio_chunks": capture.audio_chunks, "audio_duration_s": audio_s,
              "audio_total_frames": sum(x["frames"] for x in capture.audio_chunks),
              "audio_playback_stalls": capture.playback.stalls, "audio_stall_s": stall,
              "audio_max_stall_s": max((x["duration_s"] for x in capture.playback.stalls), default=0.0),
              "audio_stall_fraction_of_content": stall / audio_s if audio_s else None,
              "audio_stall_fraction_of_playback": stall / (audio_s + stall) if audio_s else None,
              "audio_decode_elapsed_s": capture.decode_elapsed_s,
              "requested_max_tokens": payload.get("max_tokens"),
              "expected_prompt_tokens": record.get("actual_processor_input_tokens"),
              "actual_prompt_tokens": len(capture.text_prompt_token_ids) if capture.text_prompt_token_ids is not None else None,
              "actual_prompt_token_ids_sha256": sha256(json.dumps(capture.text_prompt_token_ids, separators=(",", ":")).encode())
              if capture.text_prompt_token_ids is not None else None,
              "final_usage_prompt_tokens": (capture.usage or {}).get("prompt_tokens"),
              "prompt_tokens_match": (len(capture.text_prompt_token_ids) == record["actual_processor_input_tokens"]
                                      if capture.text_prompt_token_ids is not None else False)
              if "actual_processor_input_tokens" in record else None,
              "requested_modalities": payload.get("modalities", []),
              "metadata": {k: v for k, v in record.items() if k != "payload"}}
    for name in ("raw.sse", "events.jsonl"):
        path = directory / name
        result[name + "_sha256"] = sha256(path.read_bytes()) if path.exists() else None
    feedback.update(request_id, status=status, completion_monotonic_s=completion,
                    **capture.playback.feedback(completion))
    atomic_json(directory / "result.json", result)
    return result


def add_visible_token_metrics(result, tokenizer):
    """Offline canonical visible tokens, separate from model token IDs."""
    text = (Path(result["artifact_directory"]) / "text.txt").read_text(encoding="utf-8")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    arrivals = []
    for begin, end in encoded["offset_mapping"]:
        if end <= begin:
            raise ValueError("canonical text token has no visible character span")
        index = bisect.bisect_left(result["text_character_ends"], end)
        arrivals.append(result["text_chunk_arrival_monotonic_s"][index])
    result["visible_token_arrival_monotonic_s"] = arrivals
    result["visible_token_pacing"] = text_pacing(arrivals, result["scheduled_arrival_monotonic_s"])
    return result


def qualify(result):
    audio = "audio" in result["requested_modalities"]
    pacing = result.get("visible_token_pacing", result["generated_token_pacing"])
    checks = {"output_complete": result["status"] == "success",
              "ttft": result["ttft_s"] is not None and result["ttft_s"] <= 2,
              "e2e": result["e2e_s"] <= 120,
              "text_pacing": pacing["on_time_fraction"] is not None and pacing["on_time_fraction"] >= .99}
    if audio:
        checks.update({"ttfp": result["ttfp_s"] is not None and result["ttfp_s"] <= 2,
                       "audio_stall_fraction": result["audio_stall_fraction_of_content"] is not None
                       and result["audio_stall_fraction_of_content"] <= .01,
                       "audio_max_stall": result["audio_max_stall_s"] <= .1})
    result["contract_checks"] = checks
    result["contract_text_units"] = "canonical_visible_tokens" if "visible_token_pacing" in result else "model_output_token_ids"
    result["all_contracts_met"] = all(checks.values())
    return result


def replay(workload_path, output_dir, endpoint, feedback_file=None, start_delay_s=1.0,
           tokenizer=None, timeout_s=120.0):
    records = [json.loads(line) for line in Path(workload_path).read_text().splitlines() if line.strip()]
    if not records or len(records) > 128:
        raise ValueError("tradeoff pilot accepts 1 to 128 requests")
    ids = [r["source_request_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate source request ID")
    if any(not math.isfinite(r["arrival_offset_s"]) or r["arrival_offset_s"] < 0 for r in records):
        raise ValueError("invalid arrival offset")
    if start_delay_s < 0:
        raise ValueError("negative start delay")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "results.json").exists():
        raise FileExistsError("refusing to overwrite a replay")
    feedback = FeedbackWriter(feedback_file)
    origin = time.monotonic() + start_delay_s
    atomic_json(directory / "run_start.json", {"schema_version": 1, "origin_monotonic_s": origin,
                "workload_path": str(Path(workload_path).resolve()),
                "workload_sha256": sha256(Path(workload_path).read_bytes()),
                "request_count": len(records), "deadline_s": timeout_s,
                "audio_playback_policy": "start on first valid chunk; 1x; immediate resume",
                "text_pacing_units": "canonical_visible_tokens" if tokenizer else "model_output_token_ids"})

    def scheduled_send(record):
        scheduled = origin + record["arrival_offset_s"]
        remaining = scheduled - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        return send_request(record, directory, endpoint, feedback, scheduled, timeout_s)

    with ThreadPoolExecutor(max_workers=len(records)) as pool:
        futures = [pool.submit(scheduled_send, record) for record in records]
        results = [future.result() for future in futures]
    for result in results:
        if tokenizer:
            try:
                add_visible_token_metrics(result, tokenizer)
            except Exception as exc:
                result["metric_error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
                result["status"] = "error"
        qualify(result)
        atomic_json(Path(result["artifact_directory"]) / "result.json", result)
    end = max(result["client_completion_monotonic_s"] for result in results)
    span = end - origin
    passed = sum(r["all_contracts_met"] for r in results)
    success = sum(r["status"] == "success" for r in results)
    summary = {"schema_version": 1, "request_count": len(results),
               "origin_monotonic_s": origin, "last_completion_monotonic_s": end,
               "fixed_arrival_window_s": 16.0, "all_contracts_met_count": passed,
               "qualified_per_fixed_16s": passed / 16,
               "drain_span_s": span, "completion_drain_rate_rps": success / span,
               "attempts_resolved_rate_rps": len(results) / span,
               "qualified_drain_rate_rps": passed / span,
               "success_count": success,
               "text_token_count": sum(len(r["text_token_ids"]) for r in results),
               "audio_duration_s": sum(r["audio_duration_s"] for r in results),
               "max_send_lateness_s": max(r["send_lateness_s"] for r in results),
               "results": results}
    atomic_json(directory / "results.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8091/v1/chat/completions")
    parser.add_argument("--feedback-file")
    parser.add_argument("--start-delay-s", type=float, default=1.0)
    parser.add_argument("--tokenizer", help="local tokenizer directory; load before replay")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = replay(args.workload, args.output_dir, args.endpoint, args.feedback_file,
                    args.start_delay_s, tokenizer)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
