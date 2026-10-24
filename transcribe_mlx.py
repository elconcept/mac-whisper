#!/usr/bin/env python3
"""Polish speech transcription CLI (Apple Silicon MLX)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from dotenv import load_dotenv
from mac_whisper_app import DEFAULT_MODEL, FORMATS, transcribe_one
from mac_whisper_engines import ASR_BACKENDS, DIARIZATION_BACKENDS, VAD_BACKENDS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Polska transkrypcja WhisperX/MLX")
    parser.add_argument("audio", nargs="?", help="Plik albo katalog; domyślnie input/")
    parser.add_argument("--model", default=None, help=f"Model MLX (domyślnie {DEFAULT_MODEL})")
    parser.add_argument("--gcp-project", default=None,
                        help="Project ID Google Cloud (lub ADC/gcloud config)")
    parser.add_argument("--gcp-location", choices=["eu", "us"], default=None,
                        help="Domyślnie eu — Google Chirp 3")
    parser.add_argument("--asr-backend", choices=ASR_BACKENDS, default=None)
    parser.add_argument("--turbo", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vad-backend", choices=VAD_BACKENDS, default=None)
    parser.add_argument("--language", choices=["pl"], default="pl")
    parser.add_argument("--speakers", type=int)
    parser.add_argument("--min-speakers", "--min_speakers", type=int)
    parser.add_argument("--max-speakers", "--max_speakers", type=int)
    parser.add_argument("--no-diarization", action="store_true")
    parser.add_argument("--no-alignment", action="store_true")
    parser.add_argument("--no-mps-diarization", action="store_true")
    parser.add_argument("--vad", choices=["off", "on", "silero"], default=None)
    parser.add_argument("--diarize-backend", "--diarize_backend", choices=DIARIZATION_BACKENDS, default=None)
    parser.add_argument("--format", choices=FORMATS, default="all")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args(argv)
    load_dotenv()
    source = Path(args.audio or "input")
    files = sorted(
        f for f in (source.iterdir() if source.is_dir() else [source])
        if f.suffix.lower() in {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".mp4"}
    )
    if not files:
        parser.error("Brak plików audio")
    overrides = {
        "model": args.model,
        "asr_backend": args.asr_backend,
        "gcp_project": args.gcp_project,
        "gcp_location": args.gcp_location,
        "diarization_backend": args.diarize_backend,
        "vad_backend": args.vad_backend,
        "turbo": args.turbo,
        "num_speakers": args.speakers,
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers,
        "diarization": not args.no_diarization,
        "alignment": not args.no_alignment,
        "use_mps_diarization": not args.no_mps_diarization,
        "vad": args.vad != "off" if args.vad is not None else None,
        "format": args.format,
    }
    failed = 0
    for audio in files:
        print(f"Transkrypcja: {audio.name}", flush=True)
        try:
            transcribe_one(audio, args.output_dir, overrides=overrides, progress=print)
        except Exception as exc:
            print(f"ERROR [{audio.name}]: {exc}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
