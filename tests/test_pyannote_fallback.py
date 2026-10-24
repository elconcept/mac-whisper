"""Regression tests for MPS runtime and model-load failures."""
import sys
import types

import pandas as pd
import pytest


@pytest.mark.parametrize("failure_stage", ["load", "infer"])
def test_pyannote_mps_fallback_to_cpu(monkeypatch, failure_stage):
    import whisperx.pipeline as p
    import whisperx.memory_guard as memory

    monkeypatch.setattr(p.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(p.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(p, "check_memory_safety", lambda **k: True)
    monkeypatch.setattr(p, "cleanup_unified_memory", lambda **k: None)
    monkeypatch.setattr(p, "release_mlx_model", lambda: None)
    monkeypatch.setattr(p, "memory_snapshot", lambda: {})
    monkeypatch.setattr(memory, "release_mlx_model", lambda: None)
    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(
        transcribe=lambda audio, **kwargs: {
            "language": "pl", "segments": [
                {"start": 0.2, "end": 2.2, "text": "Rozmowa testowa"}
            ],
        }
    ))

    calls = []

    class FakeDiarization:
        def __init__(self, *, device, **kwargs):
            calls.append(("load", device))
            self.device = device
            if failure_stage == "load" and device == "mps":
                raise RuntimeError("MPS shader load fault")

        def __call__(self, audio, **kwargs):
            calls.append(("infer", self.device))
            if failure_stage == "infer" and self.device == "mps":
                raise RuntimeError("MPS shader inference fault")
            return pd.DataFrame([{
                "start": 0.0, "end": 3.0, "speaker": "SPEAKER_00",
            }])

    monkeypatch.setattr(p, "DiarizationPipeline", FakeDiarization)
    engine = p.TranscriptionPipeline(hf_token="test", use_mps_for_diarization=True)
    result = engine.transcribe("unused.wav", enable_alignment=False)
    assert result["segments"][0]["speaker"] == "SPEAKER_00"
    assert engine.mps_crashed is True
    assert ("load", "cpu") in calls
    assert ("infer", "cpu") in calls


def test_no_nearest_speaker_assignment_when_no_overlap():
    from whisperx.diarize import assign_word_speakers
    diarization = pd.DataFrame([{
        "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00",
    }])
    transcript = {"language": "pl", "segments": [
        {"start": 2.0, "end": 3.0, "text": "Niepewna wypowiedź"}
    ]}
    output = assign_word_speakers(diarization, transcript, fill_nearest=False)
    assert "speaker" not in output["segments"][0]
