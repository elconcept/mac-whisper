#!/usr/bin/env python3

import json
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print(f"Użycie: {Path(sys.argv[0]).name} plik.json")
    sys.exit(1)

json_file = Path(sys.argv[1])

with json_file.open("r", encoding="utf-8") as f:
    data = json.load(f)

def ts(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

out = [
    "# Transkrypcja rozmowy",
    ""
]

last_speaker = None

for seg in data.get("segments", []):
    speaker = seg.get("speaker", "UNKNOWN")
    start = ts(seg.get("start", 0))
    text = " ".join(seg.get("text", "").split())

    if not text:
        continue

    if speaker != last_speaker:
        out.append("")
        out.append("---")
        out.append("")
        out.append(f"### {speaker} [{start}]")
        out.append("")
        last_speaker = speaker

    out.append(text)
    out.append("")

md_file = json_file.with_suffix(".md")
md_file.write_text("\n".join(out), encoding="utf-8")

print(f"Zapisano: {md_file}")