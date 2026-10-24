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

from whisperx.utils.memory_guard import cleanup_unified_memory, get_free_ram_gb
from whisperx.backends.diarization_backend import DiarizationBackendFactory


class TranscriptionWorker(QThread):
    """Worker thread for transcription to avoid blocking GUI."""

    progress = Signal(int, str)  # Progress (0-100, message)
    finished = Signal(dict)  # Results
    error = Signal(str)  # Errors

    def __init__(
        self,
        audio_path: str,
        language: str = "pl",
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        initial_prompt: Optional[str] = None,
        enable_diarization: bool = True,
        enable_alignment: bool = True,
        diarization_backend: str = "pyannote",
        hf_token: Optional[str] = None,
    ):
        super().__init__()
        self.audio_path = audio_path
        self.language = language
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.initial_prompt = initial_prompt
        self.enable_diarization = enable_diarization
        self.enable_alignment = enable_alignment
        self.diarization_backend = diarization_backend
        self.hf_token = hf_token

    def run(self):
        try:
            # Check memory safety
            if get_free_ram_gb() < 2.0:
                self.progress.emit(0, "Niski poziom pamięci, czyszczenie buforów...")
                cleanup_unified_memory(force=True)

            self.progress.emit(5, "Inicjalizacja pipeline'u...")

            # Import here to avoid blocking GUI on import
            from whisperx.pipeline import TranscriptionPipeline

            pipeline = TranscriptionPipeline(
                asr_model="large-v2",
                device="cpu",  # CPU for macOS (no CUDA)
                hf_token=self.hf_token,
                language=self.language,
                batch_size=8,
                use_mps_for_diarization=True,
                diarization_backend="mlx_nemotron" if self.diarization_backend == "MLX (Nemotron - Ultra szybki)" else "pyannote",
            )

            self.progress.emit(10, "Krok 1/4: Transkrypcja ASR...")

            result = pipeline.transcribe(
                self.audio_path,
                language=self.language,
                batch_size=8,
                num_speakers=self.num_speakers,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
                enable_diarization=False,  # Diarization handled separately
                enable_alignment=False,  # Alignment handled separately
            )

            # CRITICAL: Unload ASR model before diarization to prevent OOM
            del pipeline.model
            pipeline.model = None
            cleanup_unified_memory(force=True)

            self.progress.emit(40, "Krok 2/4: Wyrównywanie czasu (Alignment)...")

            if self.enable_alignment:
                from whisperx.alignment import load_align_model, align

                align_model, align_metadata = load_align_model(
                    language_code=result["language"],
                    device="cpu",
                )

                result = align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    self.audio_path,
                    "cpu",
                )

                del align_model
                align_model = None
                cleanup_unified_memory(force=True)

            self.progress.emit(70, "Krok 3/4: Diaryzacja...")

            if self.enable_diarization:
                # Create diarization backend based on selection
                diarize_backend = DiarizationBackendFactory.create(
                    backend_type=self.diarization_backend,
                    hf_token=self.hf_token,
                    device="mps",
                )

                diarize_df = diarize_backend.diarize(
                    self.audio_path,
                    num_speakers=self.num_speakers,
                    min_speakers=self.min_speakers,
                    max_speakers=self.max_speakers,
                )

                # Assign speakers
                from whisperx.diarize import assign_word_speakers
                result = assign_word_speakers(diarize_df, result)

                # CRITICAL: Unload diarization model immediately
                diarize_backend.unload()
                cleanup_unified_memory(force=True)

            self.progress.emit(100, "Zakończono!")
            self.finished.emit(result)

        except Exception as e:
            cleanup_unified_memory(force=True)
            # Provide helpful error message for Nemotron 401 errors
            if "401" in str(e) or "Repository Not Found" in str(e):
                self.error.emit(
                    "Model MLX Nemotron jest obecnie niedostępny w Hugging Face. "
                    "Przełącz na 'Pyannote 3.1' w menu wyboru silnika diaryzacji."
                )
            else:
                self.error.emit(str(e))


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

        # Diarization engine switcher
        diarization_group = QGroupBox("Silnik diaryzacji")
        diarization_layout = QVBoxLayout()

        self.diarization_selector = QComboBox()
        self.diarization_selector.addItems([
            "Pyannote 3.1 (PyTorch - Maksymalna precyzja)",
            "MLX (Nemotron - Ultra szybki - w przygotowaniu)"
        ])
        self.diarization_selector.setCurrentIndex(0)  # Pyannote as default
        diarization_layout.addWidget(QLabel("Wybierz silnik:"))
        diarization_layout.addWidget(self.diarization_selector)

        diarization_group.setLayout(diarization_layout)
        main_layout.addWidget(diarization_group)

        # Speakers configuration
        speakers_group = QGroupBox("Konfiguracja mówców")
        speakers_layout = QVBoxLayout()

        # Speaker count
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Min mówców:"))
        self.spin_min = QSpinBox()
        self.spin_min.setMinimum(1)
        self.spin_min.setMaximum(10)
        self.spin_min.setValue(2)
        count_layout.addWidget(self.spin_min)

        count_layout.addWidget(QLabel("Max mówców:"))
        self.spin_max = QSpinBox()
        self.spin_max.setMinimum(1)
        self.spin_max.setMaximum(10)
        self.spin_max.setValue(3)
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
                elif "max_speakers" in config:
                    self.spin_max.setValue(config["max_speakers"])

                # Load speaker map
                self.speaker_map = config.get("speaker_map", {})

            except Exception as e:
                print(f"Error loading config: {e}")
                self.speaker_map = {}
        else:
            self.speaker_map = {}

    def update_token_count(self):
        """Update token count for initial prompt."""
        text = self.context_edit.toPlainText()

        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            count = len(tokens)
        except:
            count = len(text.split())

        self.token_label.setText(f"Tokeny: {count}/200")

        if count > 200:
            self.token_label.setStyleSheet("color: red;")
        else:
            self.token_label.setStyleSheet("color: black;")

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
        """Start transcription in worker thread."""
        if not self.audio_path:
            QMessageBox.warning(self, "Błąd", "Wybierz plik audio!")
            return

        # Get initial prompt with token truncation
        initial_prompt = self.context_edit.toPlainText()
        if initial_prompt:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                tokens = encoding.encode(initial_prompt)
                if len(tokens) > 200:
                    initial_prompt = encoding.decode(tokens[:200])
            except:
                pass

        # Get diarization backend type
        diarization_text = self.diarization_selector.currentText()
        if "Pyannote" in diarization_text:
            diarization_backend = "pyannote"
        else:
            diarization_backend = "mlx_nemotron"

        # Disable UI
        self.transcribe_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText("Transkrypcja w toku...")

        # Start worker
        self.worker = TranscriptionWorker(
            audio_path=self.audio_path,
            language="pl",
            num_speakers=self.spin_max.value(),
            min_speakers=self.spin_min.value(),
            max_speakers=self.spin_max.value(),
            initial_prompt=initial_prompt if initial_prompt else None,
            enable_diarization=self.diarization_check.isChecked(),
            enable_alignment=self.alignment_check.isChecked(),
            diarization_backend=diarization_backend,
            hf_token=self.hf_token,
        )

        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_progress(self, value: int, message: str):
        """Handle progress updates."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        self.results_edit.append(message)

        # Update memory info
        self.memory_label.setText(f"RAM wolne: {get_free_ram_gb():.1f} GB")

    def on_finished(self, result: dict):
        """Handle successful transcription."""
        # Apply speaker map
        for seg in result.get("segments", []):
            raw_speaker = seg.get("speaker", "UNKNOWN")
            if raw_speaker in self.speaker_map:
                seg["speaker"] = self.speaker_map[raw_speaker]

        # Display results
        self.display_results(result)

        # Re-enable UI
        self.transcribe_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Zakończono!")

        # Save to output directory
        self.save_results(result)

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


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
