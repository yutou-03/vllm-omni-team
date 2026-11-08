"""Shared transport parsing, not shared buffer state."""
import base64
import hashlib
import io
import json
from pathlib import Path
import wave


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


def clock_domain():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


async def sse_events(content):
    # Audio events can exceed aiohttp's default line-length limit. Parse byte
    # chunks and decode complete lines so UTF-8 splits are also safe.
    pending, data = b"", []
    async for chunk in content.iter_chunked(65536):
        pending += chunk
        if len(pending) > 32*1024*1024:
            raise ValueError("SSE line exceeds 32 MiB")
        while b"\n" in pending:
            raw, pending = pending.split(b"\n", 1)
            line = raw.decode("utf-8").rstrip("\r")
            if not line:
                if data:
                    yield "\n".join(data)
                    data = []
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
    if pending:
        line = pending.decode("utf-8").rstrip("\r")
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


def audio_info(raw):
    try:
        with wave.open(io.BytesIO(raw), "rb") as audio:
            frames, rate, channels, width = (audio.getnframes(), audio.getframerate(),
                                            audio.getnchannels(), audio.getsampwidth())
            pcm = audio.readframes(frames)
            if len(pcm) != frames * channels * width:
                raise ValueError("truncated WAV")
    except wave.Error:
        import soundfile as sf
        with sf.SoundFile(io.BytesIO(raw)) as audio:
            frames, rate, channels = len(audio), audio.samplerate, audio.channels
            samples = audio.read(dtype="float32", always_2d=True)
            if len(samples) != frames:
                raise ValueError("truncated audio")
            pcm = samples.tobytes()
    if frames <= 0 or rate <= 0:
        raise ValueError("empty audio chunk")
    return frames / rate, pcm, rate, channels


class StreamDecoder:
    def __init__(self, modality, special_ids=()):
        self.modality = modality
        self.special_ids = set(special_ids)
        self.pending_text_tokens = 0
        self.text = ""
        self.text_token_ids = []
        self.audio_chunks = 0
        self.units = 0.
        self.done = False
        self.hash = hashlib.sha256()

    def event(self, event):
        if event == "[DONE]":
            self.done = True
            return 0., None
        obj = json.loads(event)
        if obj.get("error"):
            raise ValueError(str(obj["error"]))
        units, retained = 0., None
        for choice in obj.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("only n=1 supported")
            delta = choice.get("delta") or {}
            content = delta.get("content")
            is_audio = obj.get("modality") == "audio"
            audio = content if is_audio else (delta.get("audio") or {}).get("data")
            if not is_audio:
                ids = choice.get("token_ids") or []
                if any(type(i) is not int for i in ids):
                    raise ValueError("invalid text token ids")
                visible = [i for i in ids if i not in self.special_ids]
                self.text_token_ids.extend(visible)
                self.pending_text_tokens += len(visible)
                if isinstance(content, str) and content:
                    self.text += content
                    if self.modality == "text":
                        units += self.pending_text_tokens
                        retained = content.encode("utf-8")
                        self.hash.update(retained)
                    self.pending_text_tokens = 0
            if audio and self.modality == "audio":
                raw = base64.b64decode(audio, validate=True)
                duration, pcm, _, _ = audio_info(raw)
                units += duration
                retained = pcm
                self.hash.update(pcm)
                self.audio_chunks += 1
        self.units += units
        return units, retained


def special_ids(model):
    config = json.loads((Path(model) / "tokenizer_config.json").read_text())
    return [int(k) for k, v in config.get("added_tokens_decoder", {}).items() if v.get("special")]
