# MacWhisper: Apple Silicon E2E, CI and release

## Workflow

.github/workflows/macos-release.yml uses GitHub's macos-15 runner (ARM64,
Apple M1). It does not use the unsupported macos-15-arm64 label.

- PR/main: Python syntax, existing unit tests, mocked end-to-end full
  pipeline, CLI/export and Qt GUI startup tests. After success, build
  PyInstaller .app, test its startup, upload ZIP with checksum.
- workflow_dispatch with run_real_e2e=true: above plus real MLX Whisper,
  Polish Wav2Vec2 and Pyannote on Polish speech made by espeak-ng.
- Version tags v*: mandatory real model-driven E2E. Any failure or
  missing Hugging Face secret prevents both packaging and GitHub Release.
  Only an already-pushed tag is accepted, no automatic tag creation.

## Required repo secret: HF_TOKEN

In GitHub repository Settings -> Secrets and variables -> Actions, add
HF_TOKEN with Hugging Face read access. The associated HF account must
accept gated conditions for pyannote/speaker-diarization-community-1.
Do not put credentials in source files or logs.

## Local tests and package

    uv sync --frozen --extra mlx --extra gui --extra dev
    uv pip install python-dotenv psutil tiktoken pyinstaller imageio-ffmpeg
    QT_QPA_PLATFORM=offscreen uv run --no-sync python gui.py --self-test
    uv run --no-sync python -m pytest -q tests/test_macos_e2e.py tests/test_gui_e2e.py
    brew install espeak-ng
    HF_TOKEN=... uv run --no-sync python tests/e2e_real.py
    bash scripts/build_macos.sh

Local audio is not used by CI. E2E generates Polish speech synthetically.
That test checks functional integration, not WER/CER or real-world DER.

## Publishing a release

Merge after green CI, then create/push tag from main, e.g.:

    git tag v0.1.0
    git push origin v0.1.0

The macOS application has an ad-hoc signature, not an Apple Developer ID
signature or notarization. Gatekeeper may require manual permission to
launch. Bundled binary includes FFmpeg from imageio-ffmpeg; model weights
are NOT included, and will be downloaded on first use.

Standard runner has only 7 GB unified memory. Real 60+ minute stress
tests and peak memory tests still need to be run on target M4 machines.

## Privacy

Real recordings and transcripts must never be added to Git. .gitignore
excludes future files from input and output. Existing historical commits
already contain user audio and transcripts: they remain in Git history
until cleaned separately and deliberately, with backups and coordination.


## Additional Polish workflow (P0–P2)

- `python scripts/generate_polish_sample.py` creates original Polish 30-second
  `input/sample_pl_30s.ogg` plus text reference (no third-party copyright).
- Both interfaces call `mac_whisper_app.transcribe_one` with validation, model defaults,
  optional Silero VAD, sidecar JSON and speaker labels.
- Outputs include a `.manifest.json` with recording SHA-256, model parameters,
  environment and RAM snapshots, but not user identities.
- `uv sync` currently updates stale `uv.lock` at runtime because dependency
  metadata changed; commit a freshly generated lockfile on macOS after successful
  resolution to restore strict `--locked` reproducibility.
- Quality benchmark tools live in `quality_metrics.py` and docs/QUALITY_PL.md;
  WER/DER scores from authentic multispeaker recordings require verified labels.
