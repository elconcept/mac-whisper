# Benchmark polskiej mowy — główny dialog 00/01

Reference: output/sample-pl.reference.md (VERIFIED, niezmieniony)
Source audio: input/sample-pl.m4a
Priorytet: SPEAKER_00 i SPEAKER_01. SPEAKER_02 jest poboczny.
Trzeci głos nadal jest zachowany w oryginalnej transkrypcji.

UWAGA: oznaczenia w referencji mają dokładność tylko 1 sekundy
i brak czasów końca. Wyniki 00/01 są **przybliżonym WER/CER
w oknach czasowych**, a nie prawdziwym diarized WER.
Modelowe speaker ID nie służą do wybierania słów w oknach.
Zachowane są więc pomyłki w rozpoznaniu mowy i zakłócenia wewnątrz
głównych wypowiedzi; omijane są odrębne okna SPEAKER_02.
Pokrycie głównej referencji: 93.7%
(251 / 268 słów).
Pominięte sekundy ze zbieżnymi głosami: [122, 162, 172].

| Model | WER 00/01 (macro) | WER 00 | WER 01 | WER 00/01 łącznie | CER 00/01 | WER całe audio | DER approx. |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlx-audio-large-v3-turbo-mlx-nemotron-bf16 | 45.40% | 40.00% | 50.79% | 36.65% | 27.14% | 39.80% | — |
| mlx-audio-parakeet-v3-mlx-nemotron-8bit | 48.97% | 40.00% | 57.94% | 42.23% | 30.03% | 46.71% | — |
| mlx-whisper-large-v3-turbo-pyannote | 49.38% | 44.00% | 54.76% | 37.05% | 26.60% | 45.72% | — |
| mlx-whisper-large-v3-mlx-sortformer-v2 | 52.15% | 41.60% | 62.70% | 45.02% | 31.74% | 45.07% | — |
| mlx-audio-parakeet-redux-mlx-sortformer-v1 | 56.13% | 44.80% | 67.46% | 51.39% | 38.59% | 53.29% | — |

### Metryka dodatkowa: wszystkie wypowiedzi

| Model | WER całego nagrania | CER całego nagrania |
|---|---:|---:|
| mlx-audio-large-v3-turbo-mlx-nemotron-bf16 | 39.80% | 29.22% |
| mlx-audio-parakeet-v3-mlx-nemotron-8bit | 46.71% | 33.13% |
| mlx-whisper-large-v3-turbo-pyannote | 45.72% | 35.92% |
| mlx-whisper-large-v3-mlx-sortformer-v2 | 45.07% | 33.58% |
| mlx-audio-parakeet-redux-mlx-sortformer-v1 | 53.29% | 39.16% |

Speaker DER is **not calculated** until an independently manually
verified output/sample-pl.speakers.reference.json provides start/end/overlap
intervals. If supplied, the table contains a 100-ms-frame approximate
speaker-time error after speaker-label permutation, NOT pyannote's
official collar- and overlap-adjusted DER. This approximation is sensitive
to exact boundaries and should not be treated as legal evidence.
Model predictions alone do not establish individual identities.
