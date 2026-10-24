"""Model registry, adapter and transcription streaming regression tests."""
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from mac_whisper_engines import (
    ASR_BACKENDS, ASR_MODELS, DIARIZATION_BACKENDS, VAD_BACKENDS,
    model_id, normalized_segments, merge_spans, mlx_audio_transcribe,
    get_vad_spans, mlx_audio_diarization, overlap_intervals,
)


@pytest.fixture(autouse=True)
def offline_public_weights(monkeypatch):
    # These are unit tests: never reach Hugging Face. Real downloads are tested in Actions.
    monkeypatch.setattr("mac_whisper_engines.public_repo_path", lambda repo: repo)


def test_registry_exposes_all_supported_polish_modes():
    assert ASR_BACKENDS == ("mlx-whisper", "mlx-audio")
    assert "parakeet-v3" in ASR_MODELS["mlx-audio"]
    assert "parakeet-redux" in ASR_MODELS["mlx-audio"]
    assert "mlx-nemotron-8bit" in DIARIZATION_BACKENDS
    assert "mlx-nemotron-bf16" in DIARIZATION_BACKENDS
    assert "mlx-sortformer-v1" in DIARIZATION_BACKENDS
    assert "mlx-sortformer-v2" in DIARIZATION_BACKENDS
    assert VAD_BACKENDS[0] == "mlx-silero"
    assert model_id("mlx-audio", "large-v3-turbo").endswith("fp16")
    with pytest.raises(ValueError):
        model_id("unknown", "large-v3")


def test_default_app_uses_mlx_silero():
    from mac_whisper_app import Options
    opts = Options()
    assert opts.vad and opts.vad_backend == "mlx-silero"
    assert opts.diarization_backend == "pyannote"
    assert opts.asr_backend == "mlx-whisper"
    with pytest.raises(ValueError):
        Options(vad=True, vad_backend="off").validate()


def test_speech_chunks_have_original_timestamps():
    spans = [{"start": 5, "end": 7}, {"start": 7.15, "end": 9},
             {"start": 35, "end": 38}]
    result = merge_spans(spans, 40)
    assert len(result) == 2
    assert result[0][0] > 4
    assert result[1][0] >= 34
    assert merge_spans([], 40) == []


def test_mlx_audio_whisper_and_polish_parakeet(monkeypatch):
    calls = []

    class Model:
        def generate(self, path, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(
                text="Dzień dobry",
                segments=[{"start": 0, "end": 2, "text": "Dzień dobry",
                           "words": [{"word": "Dzień", "start": 0, "end": 1}]}]
            )

    model = Model()
    stt = types.ModuleType("mlx_audio.stt")
    stt.load = lambda mid, **kwargs: model
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", stt)

    obj, result = mlx_audio_transcribe("a.wav", "mlx-community/whisper-large-v3",
                                      initial_prompt="Kraków")
    assert obj is model
    assert result["segments"][0]["words"][0]["word"] == "Dzień"
    assert calls[0]["language"] == "pl"
    assert calls[0]["initial_prompt"] == "Kraków"
    mlx_audio_transcribe("a.wav", "mlx-community/parakeet-tdt-0.6b-v3",
                         model=model)
    assert "initial_prompt" not in calls[-1]


def test_mlx_silero_adapter(monkeypatch):
    class FakeVad:
        def get_speech_timestamps(self, path, **kwargs):
            assert kwargs["return_seconds"]
            return [{"start": 1., "end": 2.}]
    vad = types.ModuleType("mlx_audio.vad")
    vad.load = lambda path, **kwargs: FakeVad()
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.vad", vad)
    assert get_vad_spans("recording.wav", "mlx-silero") == [{"start": 1., "end": 2.}]


def test_nemotron_sortformer_outputs_and_overlap(monkeypatch):
    class Segment:
        def __init__(self, speaker, start, end):
            self.speaker, self.start, self.end = speaker, start, end
    class Model:
        def generate(self, *args, **kwargs):
            return types.SimpleNamespace(segments=[
                Segment("speaker_0", 0, 2),
                Segment("speaker_1", 1, 3),
            ])
    vad = types.ModuleType("mlx_audio.vad")
    vad.load = lambda model, **kw: Model()
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.vad", vad)
    for backend in DIARIZATION_BACKENDS[1:]:
        df = mlx_audio_diarization("recording.wav", backend)
        assert df.speaker.tolist() == ["SPEAKER_00", "SPEAKER_01"]
        overlap = overlap_intervals(df)
        assert overlap[0]["start"] == 1.0 and overlap[0]["end"] == 2.0


def test_overlapping_tracks_same_speaker_not_ambiguity():
    df = pd.DataFrame([
        {"start": 0, "end": 2, "speaker": "SPEAKER_00"},
        {"start": 1, "end": 3, "speaker": "SPEAKER_00"},
    ])
    assert overlap_intervals(df) == []


def test_streaming_gui_controls_and_turbo(monkeypatch):
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    from gui import MainWindow
    monkeypatch.setenv("HF_TOKEN", "mock-local")
    app = QApplication.instance() or QApplication([])
    widget = MainWindow()
    assert widget.vad_check.isChecked()
    assert widget.vad_selector.currentText() == "mlx-silero"
    assert widget.asr_selector.currentText() == "mlx-whisper"
    widget.asr_selector.setCurrentText("mlx-audio")
    assert widget.model_selector.findText("parakeet-v3") >= 0
    widget.model_selector.setCurrentText("large-v3")
    widget.turbo_check.setChecked(True)
    assert widget.model_selector.currentText() == "large-v3-turbo"
    widget.on_partial({"start": 3., "end": 4., "text": "Dzień dobry"})
    assert "Dzień" in widget.results_edit.toPlainText()
    widget.close()
