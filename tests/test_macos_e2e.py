import json
import math
import struct
import sys
import types
import wave
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def speech_wav(tmp_path):
    path = tmp_path / "rozmowa.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"".join(struct.pack("<h", int(1200 * math.sin(i / 21))) for i in range(64000)))
    return path


def test_mlx_alignment_pyannote_export_integration(speech_wav, monkeypatch, tmp_path):
    import whisperx.pipeline as p
    import whisperx.memory_guard as m

    calls = {}
    monkeypatch.setattr(p.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(p.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(p, "check_memory_safety", lambda **kwargs: True)
    monkeypatch.setattr(p, "cleanup_unified_memory", lambda **kwargs: None)
    monkeypatch.setattr(p, "release_mlx_model", lambda: None)
    monkeypatch.setattr(p, "memory_snapshot", lambda: {"available_gb": 4})
    monkeypatch.setattr(m, "release_mlx_model", lambda: None)

    def fake_transcribe(audio, **kwargs):
        calls["mlx"] = kwargs
        return {"language": "pl", "segments": [
            {"start": .1, "end": 1.5, "text": " Dzień dobry."},
            {"start": 2, "end": 3.7, "text": " Witam serdecznie."},
        ]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=fake_transcribe))
    monkeypatch.setattr(p, "load_align_model", lambda **kwargs: ("model", {"dictionary": {}}))

    def fake_align(segments, model, metadata, audio, device):
        assert device == "cpu"
        return {"language": "pl", "segments": [
            {**s, "words": [{"word": s["text"].strip(), "start": s["start"], "end": s["end"], "score": .9}]}
            for s in segments
        ], "word_segments": []}

    monkeypatch.setattr(p, "align", fake_align)

    class FakeDiarization:
        def __init__(self, **kwargs):
            calls["device"] = kwargs["device"]
        def __call__(self, audio, **kwargs):
            return pd.DataFrame([
                {"speaker": "SPEAKER_00", "start": 0, "end": 1.7},
                {"speaker": "SPEAKER_01", "start": 1.7, "end": 4},
            ])

    monkeypatch.setattr(p, "DiarizationPipeline", FakeDiarization)
    engine = p.TranscriptionPipeline(hf_token="testing", use_mps_for_diarization=False)
    result = engine.transcribe(str(speech_wav), num_speakers=2,
                               initial_prompt="Kraków, protokół.")
    engine.cleanup()
    assert calls["mlx"]["language"] == "pl"
    assert calls["mlx"]["initial_prompt"] == "Kraków, protokół."
    assert [s["speaker"] for s in result["segments"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert all("speaker" in s["words"][0] for s in result["segments"])
    assert calls["device"] == "cpu"

    from mac_whisper_app import save_outputs
    save_outputs(result, speech_wav, tmp_path, "all")
    for suffix in ("json", "md", "txt", "srt", "tsv", "vtt"):
        assert (tmp_path / f"rozmowa.{suffix}").is_file()


def test_cli_uses_shared_service(speech_wav, monkeypatch, tmp_path):
    import transcribe_mlx
    import mac_whisper_app
    called = []

    def fake_run(audio, output_dir, *, overrides, progress):
        called.append(overrides)
        return {"language": "pl", "segments": []}

    monkeypatch.setattr(mac_whisper_app, "transcribe_one", fake_run)
    assert transcribe_mlx.main([str(speech_wav), "--speakers", "2", "--no-diarization",
                                "--output-dir", str(tmp_path)]) == 0
    assert called[0]["num_speakers"] == 2
    assert called[0]["diarization"] is False


def test_corrupt_json_is_error_not_silent_loss(speech_wav):
    from mac_whisper_app import options_for_file
    speech_wav.with_suffix(".json").write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid sidecar JSON"):
        options_for_file(speech_wav)


def test_empty_recording_export(tmp_path):
    from mac_whisper_app import save_outputs
    audio = tmp_path / "silence.wav"
    audio.touch()
    save_outputs({"segments": [], "language": "pl"}, audio, tmp_path, "all")
    assert json.loads((tmp_path / "silence.json").read_text())["segments"] == []
    assert (tmp_path / "silence.md").exists()


def test_memory_fallback_and_cache_release(monkeypatch):
    from whisperx.memory_guard import check_memory_safety, release_mlx_model
    import whisperx.memory_guard as m
    monkeypatch.setattr(m, "get_free_ram_gb", lambda: 0.1)
    monkeypatch.setattr(m, "cleanup_unified_memory", lambda **kwargs: None)
    with pytest.raises(MemoryError):
        check_memory_safety(threshold_gb=1)

    fake_holder = type("ModelHolder", (), {"model": object(), "model_path": "x"})
    fake_module = types.SimpleNamespace(ModelHolder=fake_holder)
    import importlib
    monkeypatch.setattr(importlib, "import_module", lambda name: fake_module)
    release_mlx_model()
    assert fake_holder.model is None
    assert fake_holder.model_path is None


def test_speaker_range_configuration(speech_wav):
    from mac_whisper_app import options_for_file
    speech_wav.with_suffix(".json").write_text(
        json.dumps({"num_speakers": 2, "speaker_map": {"SPEAKER_00": "A"}}), encoding="utf-8"
    )
    assert options_for_file(speech_wav, {"min_speakers": 2, "max_speakers": 4}).num_speakers is None
    assert options_for_file(speech_wav).num_speakers == 2
    with pytest.raises(ValueError):
        options_for_file(speech_wav, {"min_speakers": 4, "max_speakers": 2})


def test_quality_metric_algorithms():
    from quality_metrics import wer, cer, der, word_alignment_mae
    assert wer("Ala ma kota", "Ala ma psa") == pytest.approx(1 / 3)
    assert cer("kot", "kat") == pytest.approx(1 / 3)
    ref = [{"start": 0, "end": 1, "speaker": "A"},
           {"start": 1, "end": 2, "speaker": "B"}]
    pred = [{"start": 0, "end": 1, "speaker": "1"},
            {"start": 1, "end": 2, "speaker": "2"}]
    assert der(ref, pred) == pytest.approx(0)
    assert word_alignment_mae(
        [{"word": "test", "start": 0, "end": 1}],
        [{"word": "test", "start": .1, "end": .9}]
    ) == pytest.approx(.1)
