"""Headless Qt startup and file selection smoke tests."""
import os
import subprocess
import sys
from pathlib import Path
import pytest


def test_gui_starts_headless():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", HF_TOKEN="local-test-only")
    proc = subprocess.run([sys.executable, str(root / "gui.py"), "--self-test"],
                          cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "GUI self-test OK" in proc.stdout


def test_gui_file_selection(monkeypatch, tmp_path):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication, QFileDialog
    from gui import MainWindow
    app = QApplication.instance() or QApplication([])
    audio = tmp_path / "polski.wav"
    audio.write_bytes(b"RIFFdummy")
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *a, **k: (str(audio), ""))
    widget = MainWindow()
    widget.browse_file()
    assert widget.audio_path == str(audio)
    assert widget.transcribe_btn.isEnabled()
    assert widget.model_selector.currentText() == "large-v3-turbo"
    widget.close()


def test_gui_worker_uses_shared_service(monkeypatch, tmp_path):
    import mac_whisper_app
    from gui import TranscriptionWorker
    recorded = []

    def fake_run(path, out, *, overrides, progress):
        recorded.append((path, overrides))
        progress("Zapisywanie wyników")
        return {"language": "pl", "segments": []}

    monkeypatch.setattr(mac_whisper_app, "transcribe_one", fake_run)
    worker = TranscriptionWorker(str(tmp_path / "fixture.wav"),
                                 overrides={"model": "large-v3-turbo"})
    results = []
    worker.finished.connect(lambda r: results.append(r))
    worker.run()
    assert results and results[0]["language"] == "pl"
    assert recorded[0][1]["model"] == "large-v3-turbo"
