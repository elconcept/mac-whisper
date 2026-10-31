#!/usr/bin/env python3
"""Merge transcription hypotheses, keeping actual offsets relative to source.

Speaker identities are never guessed. Keep audio untouched and ASR output
independent of any prior human transcript.
"""
import argparse
import json
import re
from pathlib import Path

def timestamp(value):
    sec = max(0, int(float(value)))
    return f"{sec//3600:02d}:{(sec//60)%60:02d}:{sec%60:02d}"

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("directory", type=Path)
    cli.add_argument("--seconds-per-part", type=float, default=600.0)
    args = cli.parse_args()
    items = []
    parts = sorted(args.directory.glob("part_*.json"))
    if not parts:
        raise SystemExit("No ASR JSON output found")
    for file in parts:
        m = re.fullmatch(r"part_(\d+)", file.stem)
        if not m:
            continue
        offset = int(m.group(1)) * args.seconds_per_part
        raw = json.loads(file.read_text(encoding="utf-8"))
        for seg in raw.get("segments", []):
            words = str(seg.get("text", "")).strip()
            if not words:
                continue
            start = float(seg.get("start", 0.0)) + offset
            end = float(seg.get("end", seg.get("start", 0.0))) + offset
            items.append({"start": start, "end": end, "text": words,
                          "chunk_index": int(m.group(1))})
    items.sort(key=lambda x: (x["start"], x["end"]))
    (args.directory / "combined.json").write_text(
        json.dumps({"language": "pl", "segments": items,
                    "speaker_identity": "not_verified"},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    md = [
        "# Stawska — niezależna transkrypcja automatyczna",
        "",
        "> Znaczniki czasu pochodzą z modelu ASR i offsetu chunku. "
        "Nie potwierdzają tożsamości mówcy ani dosłowności wypowiedzi.",
        "",
    ]
    md.extend(f"[{timestamp(s['start'])}] [NIEUSTALONY]: {s['text']}" for s in items)
    (args.directory / "combined.md").write_text(
        "\n\n".join(md) + "\n", encoding="utf-8")
    print(f"Merged {len(items)} segments across {len(parts)} clips")

if __name__ == "__main__":
    main()
