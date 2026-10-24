"""Adapters for native MLX-Audio Polish ASR, Silero VAD and diarization.

Heavy libraries load lazily only after the user selects the corresponding
backend. All returned timestamps are in the ORIGINAL audio clock.
"""
from __future__ import annotations

ASR_BACKENDS = ("mlx-whisper", "mlx-audio")
ASR_MODELS = {
    "mlx-whisper": {
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "tiny": "mlx-community/whisper-tiny-mlx",
    },
    "mlx-audio": {
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo-asr-fp16",
        "large-v3": "mlx-community/whisper-large-v3",
        "parakeet-v3": "mlx-community/parakeet-tdt-0.6b-v3",
        "parakeet-redux": "moondream/parakeet-redux",
    },
}
DIARIZATION_MODELS = {
    "mlx-nemotron-8bit": "mlx-community/Nemotron-3-Diarization-8bit",
    "mlx-nemotron-bf16": "mlx-community/Nemotron-3-Diarization",
    "mlx-sortformer-v1": "mlx-community/diar_sortformer_4spk-v1-fp16",
    "mlx-sortformer-v2": "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16",
}
DIARIZATION_BACKENDS = ("pyannote", *DIARIZATION_MODELS)
VAD_BACKENDS = ("mlx-silero", "torch-silero", "off")


def public_repo_path(repo_id: str) -> str:
    """Download public MLX weights anonymously even if local HF_TOKEN is expired.

    HF tokens are used only for gated/private models like Pyannote. Never
    forward a token to public mlx-community models.
    """
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, token=False)



def model_id(backend: str, model: str) -> str:
    if backend not in ASR_MODELS:
        raise ValueError(f"Unknown ASR backend: {backend}")
    if model in ASR_MODELS[backend]:
        return ASR_MODELS[backend][model]
    if "/" in model:
        return model
    raise ValueError(f"Unsupported model {model} for {backend}")


def _fields(obj, keys):
    return obj if isinstance(obj, dict) else {
        k: getattr(obj, k) for k in keys if hasattr(obj, k)
    }


def normalized_segments(output, duration: float) -> list[dict]:
    """Translate native Whisper/Parakeet outputs to canonical WhisperX records."""
    raw = getattr(output, "segments", None) or getattr(output, "sentences", None) or []
    segments = []
    for row in raw:
        item = _fields(row, ("text", "start", "end", "start_time", "end_time", "words"))
        start = item.get("start", item.get("start_time"))
        end = item.get("end", item.get("end_time"))
        if start is None or end is None:
            continue
        record = {"start": float(start), "end": float(end), "text": str(item.get("text", ""))}
        words = []
        for word in item.get("words") or []:
            w = _fields(word, ("word", "text", "start", "end", "start_time", "end_time"))
            a, b = w.get("start", w.get("start_time")), w.get("end", w.get("end_time"))
            if a is not None and b is not None:
                words.append({"start": float(a), "end": float(b),
                              "word": str(w.get("word", w.get("text", "")))})
        if words:
            record["words"] = words
        segments.append(record)
    if not segments and getattr(output, "text", "").strip():
        segments.append({"start": 0., "end": float(duration), "text": output.text})
    return segments


def mlx_audio_transcribe(audio, repo_id, model=None, initial_prompt=None, duration=None):
    """MLX model remains cached for adjacent clips, unloaded by pipeline after ASR."""
    from mlx_audio.stt import load
    loaded = model if model is not None else load(public_repo_path(repo_id))
    if "whisper" in repo_id.lower():
        result = loaded.generate(audio, language="pl", task="transcribe",
                                 initial_prompt=initial_prompt, word_timestamps=True)
    elif "parakeet" in repo_id.lower():
        # Parakeet v3/Redux support Polish but not Whisper-style prompting.
        result = loaded.generate(audio)
    else:
        raise ValueError(
            f"Model {repo_id} has no verified Polish language adapter; "
            "select Whisper or Parakeet TDT v3 / Redux."
        )
    return loaded, {"language": "pl",
                    "segments": normalized_segments(result, float(duration or 0))}


def get_vad_spans(path: str, backend: str) -> list[dict]:
    """Silero MLX is the default: never substitute Torch silently."""
    if backend == "mlx-silero":
        from mlx_audio.vad import load
        model = load(public_repo_path("mlx-community/silero-vad"))
        try:
            spans = model.get_speech_timestamps(path, return_seconds=True)
        finally:
            del model
    elif backend == "torch-silero":
        import torch
        from silero_vad import load_silero_vad, get_speech_timestamps
        from whisperx.audio import load_audio, SAMPLE_RATE
        model = load_silero_vad()
        try:
            spans = get_speech_timestamps(
                torch.from_numpy(load_audio(path)), model,
                sampling_rate=SAMPLE_RATE, return_seconds=True, speech_pad_ms=250,
            )
        finally:
            del model
    else:
        raise ValueError(f"Unknown VAD backend: {backend}")
    return [{"start": float(x["start"]), "end": float(x["end"])} for x in spans]


def merge_spans(spans: list[dict], duration: float, gap=0.55, pad=0.25, max_len=12.):
    """Merge contiguous speech, split longer stretches without compressing time."""
    valid = sorted((max(0., float(s["start"]) - pad),
                    min(duration, float(s["end"]) + pad))
                   for s in spans if s["end"] > s["start"])
    merged = []
    for start, end in valid:
        if merged and start - merged[-1][1] <= gap and end - merged[-1][0] <= max_len:
            merged[-1] = merged[-1][0], max(merged[-1][1], end)
        else:
            merged.append((start, end))
    clips = []
    for start, end in merged:
        while end - start > max_len:
            clips.append((start, start + max_len))
            start += max_len
        if end - start >= .15:
            clips.append((start, end))
    return clips


def full_chunks(duration: float, size=12.):
    out = []
    start = 0.
    while start < duration:
        out.append((start, min(duration, start + size)))
        start += size
    return out


def mlx_audio_diarization(path, backend):
    import pandas as pd
    from mlx_audio.vad import load
    if backend not in DIARIZATION_MODELS:
        raise ValueError(f"Unsupported diarization backend: {backend}")
    model = load(public_repo_path(DIARIZATION_MODELS[backend]), strict=True)
    try:
        output = model.generate(path, threshold=0.5)
        rows = []
        for s in output.segments:
            who = str(s.speaker)
            if who.startswith("speaker_") and who[8:].isdigit():
                who = f"SPEAKER_{int(who[8:]):02d}"
            elif who.isdigit():
                who = f"SPEAKER_{int(who):02d}"
            rows.append({"start": float(s.start), "end": float(s.end), "speaker": who})
        return pd.DataFrame(rows, columns=["start", "end", "speaker"])
    finally:
        del model


def overlap_intervals(diar_df) -> list[dict]:
    """Overlap requires >=2 distinct speakers, not two same-speaker tracks."""
    events = []
    for row in diar_df.to_dict("records"):
        if row["end"] > row["start"]:
            events.append((float(row["start"]), 1, str(row["speaker"])))
            events.append((float(row["end"]), -1, str(row["speaker"])))
    events.sort(key=lambda e: (e[0], e[1]))
    active = {}
    last = None
    out = []
    for t, delta, who in events:
        if last is not None and t > last and len(active) > 1:
            out.append({"start": last, "end": t, "speakers": sorted(active)})
        active[who] = active.get(who, 0) + delta
        if active[who] <= 0:
            del active[who]
        last = t
    return out
