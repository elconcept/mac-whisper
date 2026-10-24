#!/usr/bin/env python3
"""Archive completed GitHub Actions transcription outputs.

Input: directory with benchmark artifact subdirectories (gh run download).
Never writes/overwrites human-edited output/sample-pl.reference.md.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

def archive(download_root: Path, output_dir: Path, source_run: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = []
    for report in sorted(download_root.rglob("*.report.json")):
        data = json.loads(report.read_text(encoding="utf-8"))
        if data.get("status") != "passed":
            continue
        name = data["mode"]
        transcripts = list(report.parent.rglob("sample-pl.json"))
        if len(transcripts) != 1:
            raise ValueError(f"{name}: exactly one sample-pl.json required")
        result = json.loads(transcripts[0].read_text(encoding="utf-8"))
        if not result.get("segments"):
            raise ValueError(f"{name}: missing transcript segments")
        for src, dest in (
            (transcripts[0], output_dir / f"{name}.json"),
            (report, output_dir / f"{name}.report.json"),
        ):
            if dest.exists() and dest.read_bytes() != src.read_bytes():
                raise FileExistsError(f"Refusing to overwrite: {dest}")
            if not dest.exists():
                shutil.copyfile(src, dest)
        valid.append((name, data))
    if not valid:
        raise RuntimeError("No completed model transcription outputs")
    notes = [
        "# Polish ASR hypotheses — input/sample-pl.m4a",
        "",
        f"Provenance: GitHub Actions run {source_run}.",
        "RAW MACHINE HYPOTHESES, not verified transcripts or speaker identities.",
        "Human reference is in ../sample-pl.reference.md and is NEVER overwritten.",
        "",
        "| ASR + diarization | Predicted speakers | Segments | Runtime |",
        "|---|---:|---:|---:|",
    ]
    for name, data in valid:
        notes.append(f"| {name} | {data['observed_speaker_count']} | "
                     f"{data['segments']} | {data['elapsed_seconds']} s |")
    readme = output_dir / "README.md"
    if not readme.exists():
        readme.write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("Archived", len(valid), "complete transcripts")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("download_root", type=Path)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--output", type=Path, default=Path("output/hypotheses"))
    args = parser.parse_args()
    archive(args.download_root, args.output, args.run_id)

if __name__ == "__main__":
    main()
