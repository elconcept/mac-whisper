"""Shared Polish transcription service used by CLI and GUI.

No model imports at module import time: Qt can start before MLX/PyTorch loads.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DEFAULT_MODEL = "large-v3-turbo"
FORMATS = ("all", "json", "md", "txt", "srt", "vtt", "tsv")


@dataclass
class Options:
    model: str = DEFAULT_MODEL
    initial_prompt: str | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    diarization: bool = True
    alignment: bool = True
    use_mps_diarization: bool = True
    vad: bool = True
    asr_backend: str = "mlx-whisper"
    diarization_backend: str = "pyannote"
    vad_backend: str = "mlx-silero"
    turbo: bool = True
    format: str = "all"
    speaker_map: dict[str, str] | None = None

    def validate(self) -> None:
        from mac_whisper_engines import ASR_BACKENDS, DIARIZATION_BACKENDS, VAD_BACKENDS, model_id
        if self.asr_backend not in ASR_BACKENDS:
            raise ValueError(f"Unknown ASR backend: {self.asr_backend}")
        if self.diarization_backend not in DIARIZATION_BACKENDS:
            raise ValueError(f"Unknown diarization backend: {self.diarization_backend}")
        if self.vad_backend not in VAD_BACKENDS:
            raise ValueError(f"Unknown VAD backend: {self.vad_backend}")
        if self.vad and self.vad_backend == "off":
            raise ValueError("VAD enabled with backend=off: select a Silero backend or disable VAD")
        model_id(self.asr_backend, self.model)
        if not self.model or not isinstance(self.model, str):
            raise ValueError("ASR model must be a non-empty name")
        if self.format not in FORMATS:
            raise ValueError(f"Invalid output format: {self.format}")
        if self.initial_prompt is not None and not isinstance(self.initial_prompt, str):
            raise ValueError("initial_prompt must be a string")
        for field in ("num_speakers", "min_speakers", "max_speakers"):
            value = getattr(self, field)
            if value is not None and (type(value) is not int or not 1 <= value <= 20):
                raise ValueError(f"{field} must be an integer between 1 and 20")
        if self.num_speakers is not None and (self.min_speakers is not None or self.max_speakers is not None):
            raise ValueError("Choose either an exact number of speakers or a min/max range")
        if self.min_speakers is not None and self.max_speakers is not None and self.min_speakers > self.max_speakers:
            raise ValueError("min_speakers must be <= max_speakers")
        if self.speaker_map is not None:
            if not isinstance(self.speaker_map, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in self.speaker_map.items()
            ):
                raise ValueError("speaker_map must map strings to strings")


def load_sidecar(audio: Path) -> dict[str, Any]:
    path = audio.with_suffix(".json")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"Invalid sidecar JSON {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Sidecar {path.name} must contain a JSON object")
    return data


def options_for_file(audio: Path, overrides: dict[str, Any] | None = None) -> Options:
    """CLI/GUI explicit settings take precedence over sidecar, then defaults."""
    sidecar = load_sidecar(audio)
    defaults = Options()
    opts = {}
    overrides = overrides or {}
    for key in ("model", "initial_prompt", "num_speakers", "min_speakers",
                "max_speakers", "diarization", "alignment", "use_mps_diarization",
                "vad", "format", "speaker_map", "asr_backend",
                "diarization_backend", "vad_backend", "turbo"):
        chosen = overrides.get(key)
        opts[key] = chosen if chosen is not None else sidecar.get(key, getattr(defaults, key))
    # Exact CLI speaker count overrides sidecar speaker range and vice-versa.
    if overrides.get("num_speakers") is not None:
        opts["min_speakers"] = opts["max_speakers"] = None
    if overrides.get("min_speakers") is not None or overrides.get("max_speakers") is not None:
        opts["num_speakers"] = None
    if opts["num_speakers"] is not None:
        opts["min_speakers"] = opts["max_speakers"] = None
    config = Options(**opts)
    if (overrides.get("turbo") is not None or "turbo" in sidecar) and config.model in ("large-v3-turbo", "large-v3"):
        config.model = "large-v3-turbo" if config.turbo else "large-v3"
    config.validate()
    config.initial_prompt = limit_whisper_prompt(config.initial_prompt)
    return config


def limit_whisper_prompt(prompt: str | None, limit: int = 200) -> str | None:
    if not prompt:
        return None
    # Whisper's vocabulary is different from cl100k_base; use the library tokenizer.
    from mlx_whisper.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(multilingual=True, language="pl", task="transcribe")
    ids = tokenizer.encode(prompt)
    return tokenizer.decode(ids[:limit]) if len(ids) > limit else prompt


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def map_speakers(result: dict, names: dict[str, str] | None) -> None:
    for seg in result.get("segments", []):
        if seg.get("speaker") in (names or {}):
            seg["speaker"] = names[seg["speaker"]]
        for word in seg.get("words", []):
            if word.get("speaker") in (names or {}):
                word["speaker"] = names[word["speaker"]]


def write_markdown(result: dict, dest: Path) -> None:
    lines = ["# Transkrypcja", ""]
    previous = object()
    for seg in result.get("segments", []):
        text = " ".join(seg.get("text", "").split())
        if not text:
            continue
        who = seg.get("speaker", "NIEUSTALONY")
        if who != previous:
            t = max(0, int(seg["start"]))
            lines.extend([f"### {who} [{t//3600:02d}:{(t//60)%60:02d}:{t%60:02d}]", ""])
            previous = who
        lines.extend([text, ""])
    dest.write_text("\n".join(lines), encoding="utf-8")


def save_outputs(result: dict, audio: Path, outdir: Path, fmt: str = "all") -> None:
    from whisperx.utils import get_writer
    outdir.mkdir(parents=True, exist_ok=True)
    if not result.get("segments"):
        # WhisperX subtitle writer assumes segments[0]; valid silent file is an empty transcript.
        suffixes = ("txt", "json", "srt", "vtt", "tsv") if fmt == "all" else (fmt,)
        for suffix in suffixes:
            if suffix == "md":
                write_markdown(result, outdir / f"{audio.stem}.md")
            elif suffix == "json":
                (outdir / f"{audio.stem}.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            else:
                (outdir / f"{audio.stem}.{suffix}").write_text("", encoding="utf-8")
        if fmt == "all":
            write_markdown(result, outdir / f"{audio.stem}.md")
        return
    if fmt in ("all", "json", "txt", "srt", "vtt", "tsv"):
        get_writer(fmt, str(outdir))(result, str(audio), {
            "highlight_words": True, "max_line_count": None, "max_line_width": None
        })
    if fmt in ("all", "md"):
        write_markdown(result, outdir / f"{audio.stem}.md")


def transcribe_one(
    audio_path: str | Path, output_dir: str | Path = "output",
    *, overrides: dict[str, Any] | None = None, progress: Callable[[str], None] | None = None,
    on_segment: Callable[[dict], None] | None = None
) -> dict[str, Any]:
    audio = Path(audio_path).resolve()
    if not audio.is_file():
        raise FileNotFoundError(audio)
    opts = options_for_file(audio, overrides)
    if progress:
        progress("Start MLX ASR")
    from whisperx.pipeline import TranscriptionPipeline
    from whisperx.memory_guard import memory_snapshot
    from dotenv import dotenv_values
    hf_token = os.getenv("HF_TOKEN") or dotenv_values().get("HF_TOKEN")
    before = memory_snapshot()
    pipeline = TranscriptionPipeline(
        asr_model=opts.model, hf_token=hf_token, language="pl",
        use_mps_for_diarization=opts.use_mps_diarization,
        diarization_backend=opts.diarization_backend, asr_backend=opts.asr_backend,
        vad_backend=opts.vad_backend,
    )
    try:
        result = pipeline.transcribe(
            str(audio), language="pl", initial_prompt=opts.initial_prompt,
            num_speakers=opts.num_speakers, min_speakers=opts.min_speakers,
            max_speakers=opts.max_speakers, enable_alignment=opts.alignment,
            enable_diarization=opts.diarization, use_vad=opts.vad,
            on_segment=on_segment,
        )
    finally:
        pipeline.cleanup()
    if progress:
        progress("Zapisywanie wyników")
    map_speakers(result, opts.speaker_map)
    target = Path(output_dir).resolve()
    save_outputs(result, audio, target, opts.format)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"filename": audio.name, "bytes": audio.stat().st_size, "sha256": sha256(audio)},
        "pipeline": {"language": "pl", "model": opts.model, "vad": opts.vad,
                     "asr_backend": opts.asr_backend, "vad_backend": opts.vad_backend,
                     "turbo": opts.turbo,
                     "alignment": opts.alignment, "diarization": opts.diarization,
                     "diarization_backend": opts.diarization_backend, "min_speakers": opts.min_speakers,
                     "max_speakers": opts.max_speakers, "num_speakers": opts.num_speakers},
        "environment": {"python": sys.version.split()[0], "system": platform.platform(), "machine": platform.machine()},
        "memory": {"before": before, "after": memory_snapshot(),
                   "stages": getattr(pipeline, "last_memory", {})},
        "segments": len(result.get("segments", [])),
        "note": "Speaker labels are automated estimates; do not infer identity from them.",
    }
    (target / f"{audio.stem}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
