"""Offline tests for GCP Polish Chirp 3 (no credentials or paid API calls)."""
from pathlib import Path
import sys
import types
import pytest

from mac_whisper_gcp import (
    GoogleSpeechConfigError, application_default_credentials,
    auth_command, gcp_transcribe_clip, validate_location,
)


def test_chirp3_registers_and_validates():
    from mac_whisper_app import Options
    from mac_whisper_engines import ASR_MODELS
    assert ASR_MODELS["gcp"] == {"chirp_3": "chirp_3"}
    Options(asr_backend="gcp", model="chirp_3", gcp_location="eu").validate()
    with pytest.raises(ValueError):
        Options(asr_backend="gcp", model="large-v3").validate()
    with pytest.raises(ValueError):
        Options(asr_backend="gcp", model="chirp_3", gcp_location="bogus").validate()
    assert validate_location("eu") == "eu"


def test_missing_adc_has_actionable_error(monkeypatch):
    auth = types.ModuleType("google.auth")
    def fail(**kwargs):
        raise RuntimeError("no credentials")
    auth.default = fail
    google = types.ModuleType("google")
    google.auth = auth
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.auth", auth)
    with pytest.raises(GoogleSpeechConfigError, match="gcloud auth application-default login"):
        application_default_credentials("test-proj-123")


def test_gcp_short_audio_request_and_normalization(monkeypatch, tmp_path):
    constructor_args = []
    requests = []
    def dummy(**kwargs):
        constructor_args.append(kwargs)
        return types.SimpleNamespace(**kwargs)
    cloud = types.SimpleNamespace(
        AutoDetectDecodingConfig=dummy,
        RecognitionFeatures=dummy,
        RecognitionConfig=dummy,
        RecognizeRequest=dummy,
    )
    speech_root = types.ModuleType("google.cloud.speech_v2")
    speech_types = types.ModuleType("google.cloud.speech_v2.types")
    speech_types.cloud_speech = cloud
    speech_root.types = speech_types
    speech_root.SpeechClient = lambda **kwargs: None
    api = types.ModuleType("google.api_core.client_options")
    api.ClientOptions = lambda **kwargs: types.SimpleNamespace(**kwargs)
    google = types.ModuleType("google")
    google.__path__ = []
    google_cloud = types.ModuleType("google.cloud")
    google_cloud.__path__ = []
    api_core = types.ModuleType("google.api_core")
    api_core.__path__ = []
    for name, value in {
        "google": google, "google.cloud": google_cloud,
        "google.cloud.speech_v2": speech_root,
        "google.cloud.speech_v2.types": speech_types,
        "google.api_core": api_core, "google.api_core.client_options": api,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)

    class MockClient:
        def recognize(self, *, request, timeout):
            requests.append((request, timeout))
            a = types.SimpleNamespace(transcript="Dzień dobry.")
            b = types.SimpleNamespace(transcript="Czy to są akta?")
            return types.SimpleNamespace(results=[
                types.SimpleNamespace(alternatives=[a]),
                types.SimpleNamespace(alternatives=[b]),
            ])

    file = tmp_path / "clip.wav"
    file.write_bytes(b"RIFF" + b"\x00" * 1024)
    mock = MockClient()
    client, result = gcp_transcribe_clip(
        file, project_id="test-proj-123", credentials=object(),
        client=mock, duration=5.5, location="eu",
    )
    assert client is mock
    assert result == {"language": "pl", "segments": [{
        "start": 0.0, "end": 5.5,
        "text": "Dzień dobry. Czy to są akta?",
    }]}
    request, timeout = requests[0]
    assert request.recognizer == "projects/test-proj-123/locations/eu/recognizers/_"
    assert request.config.model == "chirp_3"
    assert request.config.language_codes == ["pl-PL"]
    assert request.config.features.enable_word_time_offsets is False
    assert timeout == 100
    with pytest.raises(ValueError, match=">=60"):
        gcp_transcribe_clip(file, project_id="test-proj-123",
                            credentials=object(), client=mock, duration=60)


def test_gcp_auth_button_flow_is_explicit(monkeypatch):
    monkeypatch.setattr("mac_whisper_gcp.find_gcloud", lambda: "/opt/homebrew/bin/gcloud")
    assert auth_command() == ["/opt/homebrew/bin/gcloud", "auth", "application-default", "login"]
    assert auth_command("test-proj-123")[-2:] == ["--project", "test-proj-123"]
    with pytest.raises(GoogleSpeechConfigError):
        auth_command("invalid/project")


def test_gcp_gui_backend_and_no_turbo(monkeypatch):
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    from gui import MainWindow
    monkeypatch.setenv("HF_TOKEN", "test-not-real")
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.asr_selector.setCurrentText("gcp")
    assert win.model_selector.currentText() == "chirp_3"
    assert win.gcp_auth_btn.isVisible() or not win.isVisible()
    assert not win.turbo_check.isEnabled()
    win.gcp_project_edit.setText("test-proj-123")
    win.close()
