#!/usr/bin/env python3
"""Regenerate an original ~30-second Polish sample in input/ (no copyright issues).

Requires espeak or espeak-ng and ffmpeg on PATH. Deterministic text, synthetic voice.
Source audio is deliberately not kept in Git; private recording folders are ignored.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile

TEXT = (
    "Dzień dobry. To jest przykładowe nagranie testowe w języku polskim. "
    "Służy do sprawdzania jakości transkrypcji na komputerach Apple Silicon. "
    "W pierwszej części odczytujemy proste zdania. Następnie pojawiają się "
    "nazwy własne, takie jak Kraków, Warszawa i Szczecin. "
    "Sprawdzamy także liczby: dwadzieścia trzy, sto czterdzieści siedem "
    "i tysiąc dziewięćset dziewięćdziesiąt osiem. "
    "To nagranie zostało wygenerowane komputerowo. "
    "Nie przedstawia rozmowy żadnej prawdziwej osoby."
)


def generate(destination: Path = Path("input/sample_pl_30s.ogg")) -> Path:
    tts = shutil.which("espeak-ng") or shutil.which("espeak")
    ffmpeg = shutil.which("ffmpeg")
    if not tts or not ffmpeg:
        raise RuntimeError("Install espeak-ng (or espeak) and ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="whisper-test-") as work:
        wav = Path(work) / "speech.wav"
        subprocess.run([tts, "-v", "pl", "-s", "160", "-w", str(wav), TEXT],
                       check=True, timeout=90)
        subprocess.run([ffmpeg, "-nostdin", "-y", "-loglevel", "error",
                        "-i", str(wav), "-t", "30", "-ac", "1", "-ar", "16000",
                        "-c:a", "libopus", "-b:a", "5k", str(destination)],
                       check=True, timeout=90)
    destination.with_suffix(".reference.txt").write_text(TEXT, encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(generate())
