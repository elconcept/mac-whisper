#!/usr/bin/env python3
"""Byte-for-byte split of the owner's original M4A into sub-100-MiB GitHub assets.

No transcoding, timestamp rewriting or modification of the source file.
Reassemble with: cat AUDIO_*.m4a.part-* > AUDIO_*.m4a
Always verify against the accompanying SHA-256 checksum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MiB = 1024 * 1024
EXPECTED_SOURCE = "AUDIO_20260719_2257_TK_Krystyna_Stawska.m4a"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for piece in iter(lambda: f.read(4 * MiB), b""):
            h.update(piece)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, help="Original unmodified recording")
    parser.add_argument("--output-dir", type=Path, default=Path("input/stawska-release"))
    parser.add_argument("--part-mib", type=int, default=75)
    args = parser.parse_args()

    source = args.audio.resolve()
    if source.name != EXPECTED_SOURCE:
        parser.error(f"Expected {EXPECTED_SOURCE}, got {source.name}")
    if not source.is_file():
        parser.error(f"Missing: {source}")
    if not 1 <= args.part_mib <= 90:
        parser.error("part-mib must be between 1 and 90")
    if args.output_dir.resolve() == source.parent:
        parser.error("Select a separate output directory to avoid touching source")

    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    size = args.part_mib * MiB
    parts = []
    whole_hash = hashlib.sha256()
    with source.open("rb") as inp:
        index = 0
        while True:
            first = inp.read(min(size, 4 * MiB))
            if not first:
                break
            name = f"{source.name}.part-{index:03d}"
            target = outdir / name
            if target.exists():
                parser.error(f"Output already exists; refusing overwrite: {target}")
            per_part = hashlib.sha256()
            count = 0
            with target.open("wb") as dst:
                block = first
                while block:
                    dst.write(block)
                    whole_hash.update(block)
                    per_part.update(block)
                    count += len(block)
                    left = size - count
                    if not left:
                        break
                    block = inp.read(min(left, 4 * MiB))
            parts.append({"name": name, "bytes": count, "sha256": per_part.hexdigest()})
            index += 1

    if len(parts) < 2:
        parser.error("Expected multiple parts; source may be incomplete")
    digest = whole_hash.hexdigest()
    if digest != sha256(source):
        raise SystemExit("Source changed during split; checksum mismatch")

    checksum_path = outdir / f"{source.name}.sha256"
    if checksum_path.exists():
        parser.error(f"Checksum already exists: {checksum_path}")
    checksum_path.write_text(f"{digest}  {source.name}\n", encoding="utf-8")
    manifest_path = outdir / f"{source.name}.parts.json"
    manifest_path.write_text(
        json.dumps({"source": source.name, "size": source.stat().st_size,
                    "sha256": digest, "part_size_mib": args.part_mib,
                    "parts": parts}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Created {len(parts)} parts of max {args.part_mib} MiB; original SHA-256: {digest}")
    for item in parts:
        print(f"  {item['name']}  {item['bytes']} bytes  sha256:{item['sha256']}")
    print(f"SHA-256 manifest: {checksum_path}")
    print("Upload all .part-* and .sha256 as GitHub Release assets BEFORE publishing the release.")


if __name__ == "__main__":
    main()
