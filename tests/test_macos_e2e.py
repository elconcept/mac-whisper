"""Deterministic end-to-end coverage of the Polish MLX -> align -> diarize -> export path.

All model inference is stubbed here: tests run without HF credentials or downloads.
The real model-driven E2E gate lives in tests/e2e_real.py.
"""
import json
import math
import os
import struct
import sys
import types
import wave
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def polish_wav(tmp_path):
    audio = tmp_path / "rozmowa.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"".join(struct.pack("<h", int(1200 * math.sin(i / 21))) for i in range(64000)))
    return audio


def test_complete_pipeline_with_stubbed_models(polish_wav, monkeypatch, tmp_path):
    import whisperx.pipeline as p
    from whisperx.utils import get_writer

    calls = {}

    def mlx_transcribe(audio, **kwargs):
        calls["mlx"] = kwargs
        return {
            "language": "pl",
            "segments": [
                {"start": 0.2, "end": 1.9, "text": " Dzień dobry."},
                {"start": 2.0, "end": 3.8, "text": " Witam serdecznie."},
            ],
        }

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=mlx_transcribe))
    monkeypatch.setattr(p.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(p.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(p, "cleanup_unified_memory", lambda **kwargs: None)
    monkeypatch.setattr(p, "check_memory_safety", lambda **kwargs: True)
    monkeypatch.setattr(p, "load_align_model", lambda **kwargs: ("align-model", {"dictionary": {}}))

    def fake_align(segments, model, metadata, audio, device):
        assert device == "cpu"
        aligned = []
        for s in segments:
            aligned.append({
                **s,
                "words": [{"word": s["text"].strip(), "start": s["start"], "end": s["end"], "score": 0.98}],
            })
        return {"language": "pl", "segments": aligned, "word_segments": []}

    monkeypatch.setattr(p, "align", fake_align)

    class FakeDiarization:
        def __init__(self, **kwargs):
            calls["diarize_device"] = kwargs["device"]

        def __call__(self, path, **kwargs):
            calls["diarize_args"] = kwargs
            return pd.DataFrame([
                {"start": 0.0, "end": 1.95, "speaker": "SPEAKER_00"},
                {"start": 1.95, "end": 4.0, "speaker": "SPEAKER_01"},
            ])

    monkeypatch.setattr(p, "DiarizationPipeline", FakeDiarization)
    engine = p.TranscriptionPipeline(asr_model="large-v3-turbo", hf_token="fake", language="pl", use_mps_for_diarization=False)
    result = engine.transcribe(str(polish_wav), num_speakers=2, initial_prompt="Nazwy własne: Kraków.")
    engine.cleanup()

    assert calls["mlx"]["initial_prompt"] == "Nazwy własne: Kraków."
    assert calls["mlx"]["language"] == "pl"
    assert calls["diarize_device"] == "cpu"
    assert calls["diarize_args"]["num_speakers"] == 2
    assert [s["speaker"] for s in result["segments"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert all(s["words"] and s["words"][0]["speaker"] for s in result["segments"])

    writer = get_writer("all", str(tmp_path))
    writer(result, str(polish_wav), {"highlight_words": False, "max_line_count": None, "max_line_width": None})
    for suffix in (".txt", ".json", ".srt", ".vtt", ".tsv"):
        assert (tmp_path / ("rozmowa" + suffix)).exists()
    assert json.loads((tmp_path / "rozmowa.json").read_text())["segments"][1]["speaker"] == "SPEAKER_01"


def test_cli_exports_and_propagates_per_file_prompt(polish_wav, monkeypatch, tmp_path):
    import transcribe_mlx as cli

    config = polish_wav.with_suffix(".json")
    config.write_text(json.dumps({
        "initial_prompt": "Czytelnia akt, protokół.",
        "speaker_map": {"SPEAKER_00": "Osoba A"},
        "num_speakers": 1,
    }), encoding="utf-8")
    calls = []

    class FakePipeline:
        def __init__(self, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            return {"language": "pl", "segments": [{
                "start": 0.1, "end": 2.0, "text": "Dzień dobry",
                "speaker": "SPEAKER_00",
                "words": [{"word": "Dzień", "start": 0.1, "end": 0.9},
                          {"word": "dobry", "start": 0.9, "end": 2.0}],
            }]}

        def cleanup(self):
            pass

    monkeypatch.setattr(cli, "TranscriptionPipeline", FakePipeline)
    monkeypatch.setattr(sys, "argv", ["transcribe_mlx.py", str(polish_wav), "--no-diarization",
                                     "--output-dir", str(tmp_path / "out")])
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cli.main()
    assert calls[0]["initial_prompt"] == "Czytelnia akt, protokół."
    assert calls[0]["num_speakers"] == 1
    assert (tmp_path / "out" / "rozmowa.md").exists()
    assert "Osoba A" in (tmp_path / "out" / "rozmowa.md").read_text()
    assert (tmp_path / "out" / "rozmowa.srt").exists()


def test_malformed_per_file_json_does_not_crash_cli(polish_wav, monkeypatch, tmp_path):
    import transcribe_mlx as cli

    polish_wav.with_suffix(".json").write_text("{not-json", encoding="utf-8")

    class FakePipeline:
        def __init__(self, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            return {"language": "pl", "segments": [{"start": 0.1, "end": 0.9, "text": "Test"}]}

        def cleanup(self):
            pass

    monkeypatch.setattr(cli, "TranscriptionPipeline", FakePipeline)
    monkeypatch.setattr(sys, "argv", ["transcribe_mlx.py", str(polish_wav), "--no-diarization",
                                     "--format", "md", "--output-dir", str(tmp_path / "out")])
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cli.main()
    assert (tmp_path / "out" / "rozmowa.md").exists()
