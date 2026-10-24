#!/usr/bin/env python3
import sys
import json
import os
import psutil
from pathlib import Path

from dotenv import load_dotenv
import tiktoken
import whispermlx
from whispermlx.diarize import DiarizationPipeline


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


def format_timestamp(seconds: float, separator: str = ",") -> str:
    """Formatowanie sekund do postaci HH:MM:SS,MS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{separator}{ms:03d}"


def truncate_prompt_by_tokens(text: str, max_tokens: int = 200) -> str:
    """
    Sprawdza liczbę tokenów w tekście za pomocą tokenizatora tiktoken.
    Jeśli tekst przekracza max_tokens, przycina go dokładnie do dozwolonej liczby tokenów.
    """
    try:
        # Używamy tokenizatora cl100k_base (standard dla nowoczesnych modeli)
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)

        if len(tokens) <= max_tokens:
            return text

        print(f"Ostrzeżenie: initial_prompt przekracza limit ({len(tokens)} > {max_tokens} tokenów). Przycinanie...")
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    except Exception:
        # Rezerwowy mechanizm prostego dzielenia po słowach w razie błędu tokenizatora
        words = text.split()
        if len(words) > max_tokens:
            return " ".join(words[:max_tokens])
        return text


def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Błąd: Brak klucza HF_TOKEN w pliku .env.")

    use_turbo = False
    if "-turbo" in sys.argv:
        use_turbo = True
        sys.argv.remove("-turbo")

    audio_files = []

    if len(sys.argv) == 2:
        file_path = Path(sys.argv[1]).resolve()
        if not file_path.exists():
            print(f"Nie znaleziono pliku: {file_path}")
            sys.exit(1)
        audio_files.append(file_path)
    elif len(sys.argv) == 1:
        input_dir = Path("input")
        if not input_dir.exists():
            print("Katalog 'input' nie istnieje. Utwórz folder 'input' i dodaj do niego nagrania.")
            sys.exit(1)

        for ext in ["*.m4a", "*.mp3", "*.wav"]:
            audio_files.extend(input_dir.glob(ext))

        if not audio_files:
            print("Nie znaleziono plików audio w katalogu 'input'.")
            sys.exit(1)
    else:
        print(f"Użycie: uv run transcribe [ścieżka_do_pliku] [-turbo]")
        sys.exit(1)

    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes / (1024 ** 3)
    print(f"Wykryto {ram_gb:.1f} GB pamięci RAM.")

    if use_turbo:
        if ram_gb < 12.0:
            print("Wybór: Model 'turbo-q4' (8 GB RAM + -turbo)...")
            model_name = "mlx-community/whisper-large-v3-turbo-q4"
        else:
            print("Wybór: Pełny model 'turbo' (16+ GB RAM + -turbo)...")
            model_name = "mlx-community/whisper-large-v3-turbo"
    else:
        if ram_gb < 12.0:
            print("Wybór: Model skwantowany 4-bitowy (8 GB RAM)...")
            model_name = "mlx-community/whisper-large-v3-mlx-4bit"
        else:
            print("Wybór: Pełny model domyślny (16+ GB RAM)...")
            model_name = "mlx-community/whisper-large-v3-mlx"

    print(f"Ładowanie modelu: {model_name}...")
    model = whispermlx.load_model(model_name, device="mps")
    diarize_model = DiarizationPipeline(token=hf_token, device="cpu")

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Liczba plików do przetworzenia: {len(audio_files)}")

    for audio_file in audio_files:
        print("\n" + "=" * 60)
        print(f"Analiza pliku: {audio_file.name}")
        print("=" * 60)

        initial_prompt_text = ""
        speaker_map = {}
        config_file = audio_file.with_suffix(".json")

        if config_file.exists():
            print(f"Wczytywanie pliku konfiguracyjnego: {config_file.name}")
            try:
                with open(config_file, "r", encoding="utf-8") as cf:
                    config_data = json.load(cf)
                    raw_prompt = config_data.get("initial_prompt", "")
                    # Przeliczanie i przycinanie tekstu na podstawie tokenów (limit: 200 tokenów)
                    initial_prompt_text = truncate_prompt_by_tokens(raw_prompt, max_tokens=200)
                    speaker_map = config_data.get("speaker_map", {})
            except json.JSONDecodeError:
                print(f"Błąd parsowania pliku JSON: {config_file.name}.")

        base_name = audio_file.stem
        out_live = output_dir / f"{base_name}_live_stream.txt"
        out_json = output_dir / f"{base_name}.json"
        out_srt = output_dir / f"{base_name}.srt"
        out_vtt = output_dir / f"{base_name}.vtt"
        out_tsv = output_dir / f"{base_name}.tsv"
        out_md = output_dir / f"{base_name}.md"

        transcribe_args = {
            "language": "pl",
            "beam_size": 5,
            "verbose": True
        }
        if initial_prompt_text:
            transcribe_args["initial_prompt"] = initial_prompt_text

        print(f"\nTranskrypcja w toku (logi w {out_live.name})...")
        print("-" * 50)
        with DualLogger(out_live):
            result = model.transcribe(str(audio_file), **transcribe_args)
        print("-" * 50)

        print("Wyrównywanie czasu (alignment)...")
        model_a, metadata = whispermlx.load_align_model(language_code=result["language"], device="cpu")
        result = whispermlx.align(result["segments"], model_a, metadata, str(audio_file), device="cpu")

        print("Diaryzacja (rozpoznawanie mówców)...")
        diarize_segments = diarize_model(str(audio_file))

        print("Łączenie tekstu z mówcami...")
        result = whispermlx.assign_word_speakers(diarize_segments, result)

        print("Zapisywanie plików wynikowych...")
        with open(out_json, "w", encoding="utf-8") as f_json:
            json.dump(result, f_json, ensure_ascii=False, indent=4)

        with open(out_srt, "w", encoding="utf-8") as f_srt, \
             open(out_vtt, "w", encoding="utf-8") as f_vtt, \
             open(out_tsv, "w", encoding="utf-8") as f_tsv:

            f_vtt.write("WEBVTT\n\n")
            f_tsv.write("start\tend\tspeaker\ttext\n")

            for i, seg in enumerate(result["segments"], start=1):
                start_sec = float(seg["start"])
                end_sec = float(seg["end"])

                raw_speaker = seg.get("speaker", "UNKNOWN")
                speaker = speaker_map.get(raw_speaker, raw_speaker)
                text = seg["text"].strip()

                f_tsv.write(f"{start_sec:.3f}\t{end_sec:.3f}\t{speaker}\t{text}\n")

                srt_start = format_timestamp(start_sec, ",")
                srt_end = format_timestamp(end_sec, ",")
                f_srt.write(f"{i}\n{srt_start} --> {srt_end}\n[{speaker}]: {text}\n\n")

                vtt_start = format_timestamp(start_sec, ".")
                vtt_end = format_timestamp(end_sec, ".")
                f_vtt.write(f"{vtt_start} --> {vtt_end}\n[{speaker}]: {text}\n\n")

        md_lines = ["# Transkrypcja nagrania", ""]
        last_speaker = None

        for seg in result["segments"]:
            raw_speaker = seg.get("speaker", "UNKNOWN")
            speaker = speaker_map.get(raw_speaker, raw_speaker)

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

    print("\nSukces! Operacja została zakończona.")


if __name__ == "__main__":
    main()