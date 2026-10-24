"""Regression tests: preserve upstream WhisperX pipeline, swapping only MLX ASR.

No model downloads and no credentials. A separately gated macOS E2E runs
real WhisperX VAD + MLX Turbo + Polish Wav2Vec2 alignment + Pyannote MPS.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest


def test_production_uses_best_verified_vad_with_whisperx_alignment_and_diarization():
    from mac_whisper_app import Options
    from mac_whisper_engines import VAD_BACKENDS
    opts = Options()
    assert opts.model == "large-v3-turbo"
    assert opts.asr_backend == "mlx-whisper"
    assert opts.vad
    assert opts.vad_backend == "mlx-silero"
    assert opts.diarization and opts.diarization_backend == "pyannote"
    assert opts.alignment and opts.use_mps_diarization
    assert VAD_BACKENDS[0] == "mlx-silero"


def test_whisperx_vad_uses_original_preprocess_and_merge(monkeypatch):
    from whisperx import mlx_adapter as adapter
    calls = []
    class FakeVad:
        def preprocess_audio(self, waveform):
            calls.append("upstream_preprocess")
            return waveform[None]
        def __call__(self, waveform):
            assert waveform["sample_rate"] == 16000
            calls.append("upstream_vad_inference")
            return ["unmerged-speech"]
        def merge_chunks(self, vad_result, chunk_size, onset, offset):
            assert vad_result == ["unmerged-speech"]
            assert (chunk_size, onset, offset) == (12, .5, .363)
            calls.append("upstream_merge_chunks")
            return [{"start": 1.0, "end": 3.0, "segments": [(1., 3.)]},
                    {"start": 4., "end": 5., "segments": [(4., 5.)]}]
    monkeypatch.setattr(adapter, "load_whisperx_vad", lambda *a, **kw: FakeVad())
    spans = adapter.whisperx_speech_chunks(np.zeros(16000*6, dtype=np.float32), "whisperx-pyannote")
    assert [s["start"] for s in spans] == [1.0, 4.0]
    assert calls == ["upstream_preprocess", "upstream_vad_inference", "upstream_merge_chunks"]


def test_whisperx_mlx_adapter_does_not_trust_decoder_word_timing(monkeypatch):
    from whisperx import mlx_adapter as adapter
    sample = np.zeros(16000*7, dtype=np.float32)
    monkeypatch.setattr(adapter, "whisperx_speech_chunks", lambda *a, **kw: [
        {"start": 1., "end": 3., "segments": [(1, 3)]},
        {"start": 5., "end": 6., "segments": [(5, 6)]}])
    monkeypatch.setattr("whisperx.audio.load_audio", lambda *a: sample)
    events = []
    calls = []
    def fake_mlx_decoder(filename):
        from scipy.io import wavfile
        rate, data = wavfile.read(filename)
        calls.append((rate, len(data), filename))
        return {"language": "pl", "segments": [
            {"text": "Dzień", "start": 1000, "end": 2000},
            {"text": " dobry", "start": 0, "end": 100, "words": [
                {"word": "błędne", "start": 555, "end": 777}]}]}
    result = adapter.transcribe_whisperx_mlx(
        "original.m4a", fake_mlx_decoder, vad_backend="whisperx-pyannote",
        on_segment=lambda row: events.append(row))
    assert [x["text"] for x in result["segments"]] == ["Dzień dobry", "Dzień dobry"]
    assert [(x["start"], x["end"]) for x in result["segments"]] == [(1., 3.), (5., 6.)]
    assert not any("words" in x for x in result["segments"])
    assert len(events) == 2
    assert [(r, n) for r,n,_ in calls] == [(16000,32000),(16000,16000)]


def test_default_pipeline_selects_upstream_whisperx_vad_and_local_alignment(monkeypatch):
    import whisperx.pipeline as p
    monkeypatch.setattr(p.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(p.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(p, "check_memory_safety", lambda **k: True)
    monkeypatch.setattr(p, "memory_snapshot", lambda: {})
    monkeypatch.setattr(p, "cleanup_unified_memory", lambda **k: None)
    monkeypatch.setattr(p, "release_mlx_model", lambda: None)
    calls = []
    def fake_adapter(audio, decoder, *, vad_backend, vad_device, on_segment):
        assert vad_backend == "whisperx-pyannote"
        assert vad_device == "mps"
        calls.append("whisperx_vad_merge")
        output = decoder("chunk.wav")
        return {"language": "pl", "segments": [
            {"start": 2., "end": 4., "text": output["segments"][0]["text"]}]}
    monkeypatch.setattr("whisperx.mlx_adapter.transcribe_whisperx_mlx", fake_adapter)
    def fake_mlx_decoder(self, audio, **kwargs):
        assert kwargs["without_timestamps"] is True
        assert kwargs["condition_on_previous_text"] is False
        assert kwargs["temperature"] == 0.0
        assert "beam_size" not in kwargs  # mlx-whisper has no beam-search decoder
        calls.append("mlx_decoder")
        return {"language": "pl", "segments": [
            {"start": 0, "end": 1, "text": "Protokół"}]}
    monkeypatch.setattr(p.TranscriptionPipeline, "transcribe_mlx", fake_mlx_decoder)
    def fake_load(**kwargs):
        calls.append(("w2v_load", kwargs["device"]))
        return "align_model", {"language":"pl"}
    def fake_align(segments, model, metadata, audio, device):
        calls.append(("w2v_align", device))
        return {"language":"pl","segments":[
            {**x,"words":[{"word":"Protokół","start":2.2,"end":3.8,"score":.99}]} for x in segments]}
    monkeypatch.setattr(p, "load_align_model", fake_load)
    monkeypatch.setattr(p, "align", fake_align)
    class FakeDiar:
        def __init__(self, *,device,**kwargs):
            calls.append(("pyannote_load",device))
        def __call__(self, audio, **kwargs):
            calls.append("pyannote_infer")
            return pd.DataFrame([{"start": 2.,"end":4.,"speaker":"SPEAKER_00"}])
    monkeypatch.setattr(p, "DiarizationPipeline", FakeDiar)
    engine=p.TranscriptionPipeline(vad_backend="whisperx-pyannote",
                                    diarization_backend="pyannote")
    result=engine.transcribe("fake.wav", use_vad=True)
    assert calls == ["whisperx_vad_merge","mlx_decoder",("w2v_load","mps"),
                    ("w2v_align","mps"),("pyannote_load","mps"),"pyannote_infer"]
    assert result["segments"][0]["words"][0]["start"] == 2.2
    assert result["segments"][0]["words"][0]["speaker"] == "SPEAKER_00"
    engine.cleanup()


def test_original_core_on_silence_does_not_hallucinate(monkeypatch):
    from whisperx import mlx_adapter as adapter
    monkeypatch.setattr(adapter, "whisperx_speech_chunks", lambda *a, **k: [])
    monkeypatch.setattr("whisperx.audio.load_audio", lambda *a: np.zeros(16000))
    result=adapter.transcribe_whisperx_mlx("silence.wav",
        lambda path: pytest.fail("decoder must not run on silence"),
        vad_backend="whisperx-pyannote")
    assert result["segments"] == []
