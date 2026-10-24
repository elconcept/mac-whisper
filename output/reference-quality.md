# Benchmark polskiej mowy — odniesienie po ręcznej korekcie

Reference file: output/sample-pl.reference.md
Source audio: input/sample-pl.m4a
WER/CER use the complete reference transcript, not duration-normalized excerpts.

| ASR and diarization | WER | CER | DER approximation |
|---|---:|---:|---:|
| mlx-audio-large-v3-turbo-mlx-nemotron-bf16 | 39.67% | 28.82% | — |
| mlx-whisper-large-v3-mlx-sortformer-v2 | 44.67% | 32.65% | — |
| mlx-whisper-large-v3-turbo-pyannote | 46.33% | 36.01% | — |
| mlx-audio-parakeet-v3-mlx-nemotron-8bit | 46.67% | 32.49% | — |
| mlx-audio-parakeet-redux-mlx-sortformer-v1 | 52.67% | 38.46% | — |

Speaker DER is **not calculated** until an independently manually
verified output/sample-pl.speakers.reference.json provides start/end/overlap
intervals. If supplied, the table contains a 100-ms-frame approximate
speaker-time error after speaker-label permutation, NOT pyannote's
official collar- and overlap-adjusted DER. This approximation is sensitive
to exact boundaries and should not be treated as legal evidence.
Model predictions alone do not establish individual identities.
