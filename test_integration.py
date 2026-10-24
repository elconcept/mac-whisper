#!/usr/bin/env python3
"""
Integration test for WhisperX pipeline.

Tests the full pipeline: VAD → ASR → Align → Diarize
on a 30-second segment of audio.
"""

import sys
import os
from pathlib import Path
import json
from dotenv import load_dotenv

from whisperx.pipeline import TranscriptionPipeline


def test_pipeline():
    """Test the full transcription pipeline."""
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Błąd: Brak HF_TOKEN w .env")
        return False

    # Find test audio
    input_dir = Path("input")
    audio_files = list(input_dir.glob("*.m4a")) + list(input_dir.glob("*.mp3")) + list(input_dir.glob("*.wav"))

    if not audio_files:
        print("Błąd: Brak plików audio w katalogu input/")
        return False

    audio_file = audio_files[0]
    print(f"Testowanie na pliku: {audio_file.name}")

    # Initialize pipeline
    print("\n1. Inicjalizacja pipeline'u...")
    pipeline = TranscriptionPipeline(
        asr_model="large-v2",
        device="cpu",  # Use CPU for testing (no CUDA on macOS)
        hf_token=hf_token,
        language="pl",
        batch_size=8,  # Smaller batch for testing
        use_mps_for_diarization=True,  # Test MPS with fallback
    )

    # Run transcription
    print("\n2. Uruchamianie transkrypcji (ASR + VAD)...")
    try:
        result = pipeline.transcribe(
            str(audio_file),
            language="pl",
            batch_size=8,
            num_speakers=2,  # Assume 2 speakers for test
            enable_diarization=True,
            enable_alignment=True,
        )
        print("✓ ASR + VAD: OK")
    except Exception as e:
        print(f"✗ ASR + VAD: FAILED - {e}")
        pipeline.cleanup()
        return False

    # Check results
    print("\n3. Weryfikacja wyników...")

    # Debug: Check diarization result structure
    if "diarization" in result:
        print(f"  Diarization result type: {type(result['diarization'])}")
        if hasattr(result['diarization'], 'itertracks'):
            print(f"  Diarization has itertracks method")
            tracks = list(result['diarization'].itertracks(yield_label=True))
            print(f"  Number of diarization tracks: {len(tracks)}")
            if tracks:
                print(f"  Sample track: {tracks[0]}")

    segments = result.get("segments", [])
    if not segments:
        print("✗ Brak segmentów transkrypcji")
        pipeline.cleanup()
        return False

    print(f"✓ Liczba segmentów: {len(segments)}")

    # Check for alignment
    has_alignment = any("words" in seg for seg in segments)
    if has_alignment:
        print("✓ Alignment: OK (word-level timestamps present)")
    else:
        print("⚠ Alignment: Brak word-level timestamps")

    # Check for diarization
    has_speakers = any("speaker" in seg for seg in segments)
    if has_speakers:
        speakers = set(seg.get("speaker", "UNKNOWN") for seg in segments if "speaker" in seg)
        print(f"✓ Diarization: OK (speakers: {speakers})")
    else:
        print("⚠ Diarization: Brak etykiet mówców")

    # Print sample transcript
    print("\n4. Przykładowa transkrypcja (pierwsze 3 segmenty):")
    for i, seg in enumerate(segments[:3]):
        speaker = seg.get("speaker", "UNKNOWN")
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip()
        print(f"  [{start:.2f}s - {end:.2f}s] {speaker}: {text}")

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{audio_file.stem}_test.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n5. Wyniki zapisane w: {output_file}")

    # Cleanup
    pipeline.cleanup()

    print("\n✓ Test integracyjny zakończony pomyślnie!")
    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
