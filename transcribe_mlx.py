#!/usr/bin/env python3
"""
Simplified transcription CLI using WhisperX with MPS fallback.

This script uses existing WhisperX components with automatic MPS-to-CPU
fallback for diarization on Apple Silicon.
"""

import sys
import json
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
import tiktoken

from whisperx.pipeline import TranscriptionPipeline
from whisperx.utils import get_writer
from whisperx.backends.diarization_backend import DiarizationBackendFactory


class DualLogger:
    """Równoległy zapis wyjścia do terminala oraz pliku podglądu na żywo."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "w", encoding="utf-8")

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.terminal
        self.log_file.close()

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


def truncate_prompt_by_tokens(text: str, max_tokens: int = 200) -> str:
    """
    Sprawdza liczbę tokenów w tekście za pomocą tokenizatora tiktoken.
    Jeśli tekst przekracza max_tokens, przycina go dokładnie do dozwolonej liczby tokenów.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)

        if len(tokens) <= max_tokens:
            return text

        print(f"Ostrzeżenie: initial_prompt przekracza limit ({len(tokens)} > {max_tokens} tokenów). Przycinanie...")
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    except Exception:
        words = text.split()
        if len(words) > max_tokens:
            return " ".join(words[:max_tokens])
        return text


def main():
    parser = argparse.ArgumentParser(
        description="Transkrypcja audio z WhisperX (z MPS fallback dla diaryzacji)"
    )
    parser.add_argument(
        "audio",
        nargs="?",
        help="Ścieżka do pliku audio (jeśli nie podano, przetwarza katalog 'input/')"
    )
    parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help="Model ASR (domyślnie: large-v3-turbo)"
    )
    parser.add_argument(
        "--language",
        default="pl",
        help="Język (domyślnie: pl)"
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Wyłącz diaryzację (rozpoznawanie mówców)"
    )
    parser.add_argument(
        "--no-alignment",
        action="store_true",
        help="Wyłącz wyrównywanie czasu (word-level timestamps)"
    )
    parser.add_argument(
        "--diarize-backend",
        default="pyannote",
        choices=["pyannote", "nemotron"],
        help="Silnik diaryzacji (domyślnie: pyannote)"
    )
    parser.add_argument(
        "--speakers",
        type=int,
        help="Dokładna liczba mówców"
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        help="Minimalna liczba mówców"
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Maksymalna liczba mówców"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Katalog wyjściowy (domyślnie: output)"
    )
    parser.add_argument(
        "--format",
        default="all",
        choices=["all", "srt", "vtt", "tsv", "md", "json"],
        help="Format wyjściowy (domyślnie: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size dla ASR (domyślnie: 16)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Urządzenie ASR (auto dla Apple Silicon)"
    )
    parser.add_argument(
        "--no-mps-diarization",
        action="store_true",
        help="Nie używaj MPS dla diaryzacji (zawsze CPU)"
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token and not args.no_diarization and args.diarize_backend == "pyannote":
        print("Błąd: Brak klucza HF_TOKEN dla diaryzacji Pyannote.")
        sys.exit(1)

    # Collect audio files
    audio_files = []
    if args.audio:
        file_path = Path(args.audio).resolve()
        if not file_path.exists():
            print(f"Nie znaleziono pliku: {file_path}")
            sys.exit(1)
        audio_files.append(file_path)
    else:
        input_dir = Path("input")
        if not input_dir.exists():
            print("Katalog 'input' nie istnieje. Utwórz folder 'input' i dodaj do niego nagrania.")
            sys.exit(1)

        for ext in ["*.m4a", "*.mp3", "*.wav"]:
            audio_files.extend(input_dir.glob(ext))

        if not audio_files:
            print("Nie znaleziono plików audio w katalogu 'input'.")
            sys.exit(1)

    # Initialize pipeline
    print("Inicjalizacja pipeline'u WhisperX...")
    pipeline = TranscriptionPipeline(
        asr_model=args.model,
        device=args.device,
        hf_token=hf_token,
        language=args.language,
        batch_size=args.batch_size,
        use_mps_for_diarization=not args.no_mps_diarization,
        diarization_backend="mlx_nemotron" if args.diarize_backend == "nemotron" else "pyannote",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Liczba plików do przetworzenia: {len(audio_files)}")

    for audio_file in audio_files:
        print("\n" + "=" * 60)
        print(f"Analiza pliku: {audio_file.name}")
        print("=" * 60)

        # Load config
        initial_prompt_text = ""
        speaker_map = {}
        config_data = {}
        config_file = audio_file.with_suffix(".json")

        if config_file.exists():
            print(f"Wczytywanie pliku konfiguracyjnego: {config_file.name}")
            try:
                with open(config_file, "r", encoding="utf-8") as cf:
                    config_data = json.load(cf)
                    raw_prompt = config_data.get("initial_prompt", "")
                    initial_prompt_text = truncate_prompt_by_tokens(raw_prompt, max_tokens=200)
                    speaker_map = config_data.get("speaker_map", {})
            except json.JSONDecodeError:
                print(f"Błąd parsowania pliku JSON: {config_file.name}.")

        base_name = audio_file.stem
        out_live = output_dir / f"{base_name}_live_stream.txt"

        # Override speakers with CLI args if provided
        num_speakers = args.speakers
        min_speakers = args.min_speakers
        max_speakers = args.max_speakers

        if num_speakers is None:
            num_speakers = config_data.get("num_speakers")
        if min_speakers is None:
            min_speakers = config_data.get("min_speakers")
        if max_speakers is None:
            max_speakers = config_data.get("max_speakers")

        print(f"\nTranskrypcja w toku (logi w {out_live.name})...")
        print("-" * 50)
        with DualLogger(out_live):
            result = pipeline.transcribe(
                str(audio_file),
                language=args.language,
                batch_size=args.batch_size,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                enable_diarization=not args.no_diarization,
                enable_alignment=not args.no_alignment,
                initial_prompt=initial_prompt_text or None,
            )
        print("-" * 50)

        # Apply speaker map
        for seg in result["segments"]:
            raw_speaker = seg.get("speaker", "UNKNOWN")
            if raw_speaker in speaker_map:
                seg["speaker"] = speaker_map[raw_speaker]

        # Write outputs
        print("Zapisywanie plików wynikowych...")

        # Use WhisperX writer for standard formats
        if args.format in ["all", "srt", "vtt", "tsv", "json"]:
            writer = get_writer(args.format if args.format != "all" else "all", str(output_dir))
            writer(result, str(audio_file), {
                "highlight_words": False,
                "max_line_count": None,
                "max_line_width": None,
            })

        # Custom Markdown output with speaker mapping
        if args.format in ["all", "md"]:
            out_md = output_dir / f"{base_name}.md"
            md_lines = ["# Transkrypcja nagrania", ""]
            last_speaker = None

            for seg in result["segments"]:
                speaker = seg.get("speaker", "UNKNOWN")

                start_sec = float(seg["start"])
                h = int(start_sec // 3600)
                m = int((start_sec % 3600) // 60)
                s = int(start_sec % 60)
                start_str = f"{h:02d}:{m:02d}:{s:02d}"

                text = " ".join(seg.get("text", "").split())
                if not text:
                    continue

                if speaker != last_speaker:
                    md_lines.extend(["", "---", "", f"### {speaker} [{start_str}]", ""])
                    last_speaker = speaker

                md_lines.append(text)
                md_lines.append("")

            with open(out_md, "w", encoding="utf-8") as f_md:
                f_md.write("\n".join(md_lines))

    # Cleanup
    pipeline.cleanup()

    print("\nSukces! Operacja została zakończona.")


if __name__ == "__main__":
    main()
