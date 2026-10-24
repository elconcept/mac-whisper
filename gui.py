#!/usr/bin/env python3
"""
GUI dla WhisperX z threading i przełącznikiem silników diaryzacji.

Chroni przed zamrożeniem UI (UI freeze) poprzez przeniesienie
wszystkich obliczeń do wątku QThread. Zawiera przełącznik między
Pyannote (precyzja) a MLX Nemotron (szybkość).
"""

import sys
import os
import json
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QTextEdit,
    QFileDialog, QProgressBar, QListWidget, QGroupBox,
    QMessageBox, QSplitter, QCheckBox, QComboBox
)
from dotenv import load_dotenv
import tiktoken

from whisperx.memory_guard import cleanup_unified_memory, get_free_ram_gb
from mac_whisper_app import DEFAULT_MODEL
from mac_whisper_engines import ASR_BACKENDS, ASR_MODELS, DIARIZATION_BACKENDS, VAD_BACKENDS
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor


class TranscriptionWorker(QThread):
    """Single shared pipeline for Qt and CLI; heavy imports stay in worker."""
    progress = Signal(int, str)
    partial = Signal(dict)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, audio_path: str, *, overrides: dict):
        super().__init__()
        self.audio_path = audio_path
        self.overrides = overrides

    def run(self):
        try:
            from mac_whisper_app import transcribe_one
            self.progress.emit(5, "Uruchamianie polskiego potoku MLX...")
            result = transcribe_one(
                self.audio_path, "output", overrides=self.overrides,
                progress=lambda step: self.progress.emit(
                    85 if "Zapisywanie" in step else 10, step
                ),
                on_segment=lambda item: self.partial.emit(item),
            )
            self.progress.emit(100, "Transkrypcja i zapis zakończone")
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    """Main application window with diarization engine switcher."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("WhisperX MLX Studio - M4 Optimized")
        self.setMinimumSize(800, 700)

        # Load environment
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")

        if not self.hf_token:
            QMessageBox.warning(
                self,
                "Brak HF_TOKEN",
                "Brak klucza HF_TOKEN w pliku .env.\n"
                "Diaryzacja nie będzie działać."
            )

        # State
        self.audio_path = None
        self.worker = None
        self.speaker_map = {}

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # File selection
        file_group = QGroupBox("Plik audio")
        file_layout = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Wybierz plik audio...")
        self.file_edit.setReadOnly(True)
        browse_btn = QPushButton("Przeglądaj...")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(browse_btn)
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        # Backend and model selectors are actual runtime controls.
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Silnik ASR:"))
        self.asr_selector = QComboBox()
        self.asr_selector.addItems(ASR_BACKENDS)
        model_layout.addWidget(self.asr_selector)
        model_layout.addWidget(QLabel("Model:"))
        self.model_selector = QComboBox()
        self.model_selector.setEditable(True)
        self.model_selector.addItems(ASR_MODELS["mlx-whisper"])
        model_layout.addWidget(self.model_selector)
        self.turbo_check = QCheckBox("Turbo (Whisper)")
        self.turbo_check.setChecked(True)
        model_layout.addWidget(self.turbo_check)
        main_layout.addLayout(model_layout)

        diar_layout = QHBoxLayout()
        diar_layout.addWidget(QLabel("Diaryzacja:"))
        self.diarization_selector = QComboBox()
        self.diarization_selector.addItems(DIARIZATION_BACKENDS)
        diar_layout.addWidget(self.diarization_selector)
        diar_layout.addWidget(QLabel("VAD:"))
        self.vad_selector = QComboBox()
        self.vad_selector.addItems(VAD_BACKENDS)
        diar_layout.addWidget(self.vad_selector)
        main_layout.addLayout(diar_layout)

        self.asr_selector.currentTextChanged.connect(self.populate_models)
        self.model_selector.currentTextChanged.connect(self.update_turbo_availability)
        self.turbo_check.toggled.connect(self.toggle_turbo)

        # Speakers configuration
        speakers_group = QGroupBox("Konfiguracja mówców")
        speakers_layout = QVBoxLayout()

        # Speaker count
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Min mówców (0 = auto):"))
        self.spin_min = QSpinBox()
        self.spin_min.setMinimum(0)
        self.spin_min.setMaximum(10)
        self.spin_min.setValue(0)
        count_layout.addWidget(self.spin_min)

        count_layout.addWidget(QLabel("Max mówców (0 = auto):"))
        self.spin_max = QSpinBox()
        self.spin_max.setMinimum(0)
        self.spin_max.setMaximum(10)
        self.spin_max.setValue(0)
        count_layout.addWidget(self.spin_max)

        count_layout.addStretch()
        speakers_layout.addLayout(count_layout)

        # Speaker mapping
        speakers_layout.addWidget(QLabel("Mapowanie mówców:"))
        self.speaker_list = QListWidget()
        self.speaker_list.setMaximumHeight(120)
        speakers_layout.addWidget(self.speaker_list)

        speakers_group.setLayout(speakers_layout)
        main_layout.addWidget(speakers_group)

        # Context/Initial prompt
        context_group = QGroupBox("Kontekst (initial_prompt)")
        context_layout = QVBoxLayout()
        self.context_edit = QTextEdit()
        self.context_edit.setPlaceholderText(
            "Wprowadź słowa kluczowe, nazwy własne, imiona...\n"
            "Pomoże to modelowi w rozpoznawaniu specyficznego słownictwa."
        )
        self.context_edit.setMaximumHeight(80)
        context_layout.addWidget(self.context_edit)

        # Token counter
        self.token_label = QLabel("Tokeny: 0/200")
        context_layout.addWidget(self.token_label)

        self.context_edit.textChanged.connect(self.update_token_count)

        context_group.setLayout(context_layout)
        main_layout.addWidget(context_group)

        # Options
        options_group = QGroupBox("Opcje")
        options_layout = QHBoxLayout()

        self.diarization_check = QCheckBox("Diaryzacja (rozpoznawanie mówców)")
        self.diarization_check.setChecked(True)
        options_layout.addWidget(self.diarization_check)

        self.alignment_check = QCheckBox("Wyrównywanie czasu (word-level timestamps)")
        self.alignment_check.setChecked(True)
        options_layout.addWidget(self.alignment_check)

        self.vad_check = QCheckBox("Voice Activity Detection (antyhalucynacje)")
        self.vad_check.setChecked(True)
        options_layout.addWidget(self.vad_check)
        options_layout.addStretch()
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)

        # Transcribe button
        self.transcribe_btn = QPushButton("Transkrybuj")
        self.transcribe_btn.setEnabled(False)
        self.transcribe_btn.clicked.connect(self.start_transcription)
        main_layout.addWidget(self.transcribe_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Gotowy")
        main_layout.addWidget(self.status_label)

        # Memory info
        self.memory_label = QLabel(f"RAM wolne: {get_free_ram_gb():.1f} GB")
        main_layout.addWidget(self.memory_label)

        # Results preview
        results_group = QGroupBox("Podgląd wyników")
        results_layout = QVBoxLayout()
        self.results_edit = QTextEdit()
        self.results_edit.setReadOnly(True)
        self.results_edit.setPlaceholderText("Wyniki transkrypcji pojawią się tutaj...")
        results_layout.addWidget(self.results_edit)
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

    def populate_models(self, engine: str):
        current = self.model_selector.currentText()
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        self.model_selector.addItems(ASR_MODELS[engine])
        if current in ASR_MODELS[engine]:
            self.model_selector.setCurrentText(current)
        self.model_selector.blockSignals(False)
        self.update_turbo_availability(self.model_selector.currentText())

    def update_turbo_availability(self, model: str):
        supported = model in ("large-v3", "large-v3-turbo")
        self.turbo_check.setEnabled(supported)
        if supported:
            self.turbo_check.blockSignals(True)
            self.turbo_check.setChecked(model == "large-v3-turbo")
            self.turbo_check.blockSignals(False)

    def toggle_turbo(self, enabled: bool):
        if self.model_selector.currentText() in ("large-v3", "large-v3-turbo"):
            self.model_selector.setCurrentText("large-v3-turbo" if enabled else "large-v3")

    def browse_file(self):
        """Open file dialog to select audio file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz plik audio",
            "",
            "Audio files (*.m4a *.mp3 *.wav);;All files (*)"
        )

        if file_path:
            self.audio_path = file_path
            self.file_edit.setText(file_path)
            self.transcribe_btn.setEnabled(True)

            # Try to load config
            self.load_config(file_path)

    def load_config(self, audio_path: str):
        """Load speaker map and context from JSON config if exists."""
        config_path = Path(audio_path).with_suffix(".json")

        self.context_edit.clear()
        self.spin_min.setValue(0)
        self.spin_max.setValue(0)
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

                # Load context
                if "initial_prompt" in config:
                    self.context_edit.setPlainText(config["initial_prompt"])

                # Load speaker count
                if "num_speakers" in config:
                    self.spin_max.setValue(config["num_speakers"])
                    self.spin_min.setValue(config["num_speakers"])
                else:
                    self.spin_min.setValue(config.get("min_speakers") or 0)
                    self.spin_max.setValue(config.get("max_speakers") or 0)

                # Load speaker map
                self.speaker_map = config.get("speaker_map", {})

            except Exception as e:
                print(f"Error loading config: {e}")
                self.speaker_map = {}
        else:
            self.speaker_map = {}

    def update_token_count(self):
        """Count prompt tokens with Whisper's multilingual tokenizer."""
        prompt = self.context_edit.toPlainText()
        try:
            from mlx_whisper.tokenizer import get_tokenizer
            tokens = get_tokenizer(multilingual=True, language="pl").encode(prompt)
            count = len(tokens)
            self.token_label.setText(f"Tokeny Whisper: {count}/200")
            self.token_label.setStyleSheet("color: red;" if count > 200 else "")
        except ImportError:
            self.token_label.setText("Licznik Whisper niedostępny")

    def get_speaker_map(self) -> dict:
        """Get speaker mapping from list."""
        speaker_map = {}
        for i in range(self.speaker_list.count()):
            item = self.speaker_list.item(i)
            text = item.text()
            if " → " in text:
                speaker_id, name = text.split(" → ", 1)
                speaker_map[speaker_id] = name
        return speaker_map

    def start_transcription(self):
        if not self.audio_path:
            QMessageBox.warning(self, "Błąd", "Wybierz plik audio!")
            return
        exact = self.spin_min.value() if self.spin_min.value() and self.spin_min.value() == self.spin_max.value() else None
        overrides = {
            "model": self.model_selector.currentText(),
            "asr_backend": self.asr_selector.currentText(),
            "diarization_backend": self.diarization_selector.currentText(),
            "vad_backend": self.vad_selector.currentText(),
            "turbo": self.turbo_check.isChecked(),
            "speaker_map": self.speaker_map,
            "initial_prompt": self.context_edit.toPlainText() or None,
            "num_speakers": exact,
            "min_speakers": (self.spin_min.value() or None) if exact is None else None,
            "max_speakers": (self.spin_max.value() or None) if exact is None else None,
            "diarization": self.diarization_check.isChecked(),
            "alignment": self.alignment_check.isChecked(),
            "vad": self.vad_check.isChecked(),
        }
        # Explicit speaker choices on the UI are authoritative; 0 retains sidecar/default.
        if exact is None and (self.spin_min.value() or self.spin_max.value()):
            overrides["num_speakers"] = None
        self.transcribe_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText("Transkrypcja w toku...")
        self.worker = TranscriptionWorker(self.audio_path, overrides=overrides)
        self.results_edit.clear()
        self.worker.progress.connect(self.on_progress)
        self.worker.partial.connect(self.on_partial)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_partial(self, segment: dict):
        """Append recognized chunks immediately, highlighting the last word."""
        cursor = self.results_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        sec = int(segment.get("start", 0))
        cursor.insertText(f"[{sec//3600:02d}:{sec//60%60:02d}:{sec%60:02d}] …: ")
        words = segment.get("words") or [
            {"word": word} for word in segment.get("text", "").split()
        ]
        for i, word in enumerate(words):
            fmt = QTextCharFormat()
            if i == len(words) - 1:
                fmt.setBackground(QColor("#ffe081"))
                fmt.setForeground(QColor("#222222"))
            cursor.insertText(str(word.get("word", "")) + " ", fmt)
        cursor.insertText("\n")
        self.results_edit.setTextCursor(cursor)
        self.results_edit.ensureCursorVisible()

    def on_progress(self, value: int, message: str):
        """Handle progress updates."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        # Progress remains in status label, separate from transcript.

        # Update memory info
        self.memory_label.setText(f"RAM dostępne: {get_free_ram_gb():.1f} GB")

    def on_finished(self, result: dict):
        """Handle successful transcription."""
        # Shared service already mapped labels at segment and word levels.
        # Display results
        self.display_results(result)

        # Re-enable UI
        self.transcribe_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Zakończono!")

        # Files and manifest are already saved atomically by the shared service.

    def on_error(self, error: str):
        """Handle transcription error."""
        QMessageBox.critical(self, "Błąd", f"Transkrypcja nieudana:\n{error}")

        # Re-enable UI
        self.transcribe_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Błąd")

    def display_results(self, result: dict):
        """Display transcription results in text edit."""
        lines = []

        for seg in result.get("segments", []):
            speaker = seg.get("speaker", "UNKNOWN")
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "").strip()

            # Format timestamp
            h = int(start // 3600)
            m = int((start % 3600) // 60)
            s = int(start % 60)
            timestamp = f"{h:02d}:{m:02d}:{s:02d}"

            lines.append(f"[{timestamp}] {speaker}: {text}")

        self.results_edit.setPlainText("\n".join(lines))

    def save_results(self, result: dict):
        """Save results to output directory."""
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_name = Path(self.audio_path).stem
        output_path = output_dir / f"{audio_name}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        self.status_label.setText(f"Zapisano: {output_path}")


def _configure_bundled_ffmpeg():
    """Allow WhisperX and mlx-whisper subprocesses to find packaged FFmpeg."""
    if getattr(sys, "frozen", False):
        runtime_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        if (runtime_dir / "ffmpeg").is_file():
            os.environ["PATH"] = str(runtime_dir) + os.pathsep + os.environ.get("PATH", "")


def main():
    _configure_bundled_ffmpeg()
    # Works both as a source and as a frozen PyInstaller executable.
    if "--self-test" in sys.argv:
        os.environ.setdefault("HF_TOKEN", "gui-smoke-test-no-inference")
        app = QApplication([sys.argv[0]])
        window = MainWindow()
        assert window.model_selector.count() >= 1
        assert window.diarization_selector.count() >= 4
        assert window.vad_check.isChecked()
        assert window.transcribe_btn is not None
        window.close()
        print("GUI self-test OK")
        return 0
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
