#!/usr/bin/env python3
"""
Integration test for MLX Nemotron diarization backend.

Tests the MLX Nemotron diarization on the same 30-second audio file.
"""

import sys
import os
from pathlib import Path
import json
from dotenv import load_dotenv

from whisperx.backends.diarization_backend import DiarizationBackendFactory


def test_nemotron_diarization():
    """Test MLX Nemotron diarization backend."""
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")

    # Find test audio
    input_dir = Path("input")
    audio_files = list(input_dir.glob("*.m4a")) + list(input_dir.glob("*.mp3")) + list(input_dir.glob("*.wav"))

    if not audio_files:
        print("Błąd: Brak plików audio w katalogu input/")
        return False

    audio_file = audio_files[0]
    print(f"Testowanie na pliku: {audio_file.name}")

    # Initialize MLX Nemotron backend
    print("\n1. Inicjalizacja MLX Nemotron backend...")
    try:
        backend = DiarizationBackendFactory.create(
            backend_type="mlx_nemotron",
            hf_token=hf_token,
        )
        print("✓ MLX Nemotron backend załadowany")
    except Exception as e:
        print(f"✗ Nie udało się załadować MLX Nemotron: {e}")
        return False

    # Run diarization
    print("\n2. Uruchamianie diaryzacji MLX Nemotron...")
    try:
        diarize_df = backend.diarize(
            str(audio_file),
            max_speakers=4,
        )
        print(f"✓ Diarization zakończona: {len(diarize_df)} segmentów")
    except Exception as e:
        print(f"✗ Diarization nieudana: {e}")
        backend.unload()
        return False

    # Check results
    print("\n3. Weryfikacja wyników...")
    print(f"   Kolumny: {list(diarize_df.columns)}")
    print(f"   Liczba segmentów: {len(diarize_df)}")

    if len(diarize_df) > 0:
        print("\n   Przykładowe segmenty:")
        for i, row in diarize_df.head(5).iterrows():
            print(f"   [{row['start']:.2f}s - {row['end']:.2f}s] {row['speaker']}")
    else:
        print("⚠ Brak segmentów diarization")

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{audio_file.stem}_nemotron.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(diarize_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"\n4. Wyniki zapisane w: {output_file}")

    # Cleanup
    backend.unload()

    print("\n✓ Test MLX Nemotron zakończony pomyślnie!")
    return True


if __name__ == "__main__":
    success = test_nemotron_diarization()
    sys.exit(0 if success else 1)
