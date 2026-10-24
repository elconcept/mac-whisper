# Benchmark: MLX Silero vs WhisperX Pyannote/Silero VAD (12/30 sekund)

Referencja (ręcznie VERIFIED): output/sample-pl.primary.reference.md
Audio: input/sample-pl.m4a; ASR: mlx-whisper large-v3-turbo;
alignment: Polish Wav2Vec2 CTC; diarization: Pyannote Community-1.

Podstawowa zmienna: sposób VAD i podziału na fragmenty.
Dwa przebiegi 30-sekundowego WhisperX pokazują zmienność inferencji.
Osobne testy 12-sekundowe porównują oryginalne Pyannote i Silero VAD.
WER/CER są przybliżone: ludzkie początki kwestii mają tylko sekundy
i nie ma zweryfikowanych czasów końca. Nie wyciągać wniosków o DER.

| Wariant | WER TK | WER urzędniczki | WER macro | CER dialogu | Pokrycie czasów słów | Słowa / segmenty | Słowa bez mówcy |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlx-whisper-large-v3-turbo-pyannote | 47.66% | 50.98% | 49.32% | 23.86% | 100.00% | 277 / 47 | 6 |
| mlx-whisper-large-v3-turbo-whisperx-vad-pyannote-r1 | 99.22% | 86.27% | 92.75% | 34.75% | 100.00% | 320 / 44 | 29 |
| mlx-whisper-large-v3-turbo-whisperx-vad-pyannote-r2 | 110.94% | 80.39% | 95.66% | 46.53% | 100.00% | 320 / 55 | 15 |
| mlx-whisper-large-v3-turbo-whisperx-vad-pyannote-12s | 45.31% | 70.59% | 57.95% | 35.15% | 100.00% | 180 / 25 | 2 |
| mlx-whisper-large-v3-turbo-whisperx-silero-12s-pyannote | 60.94% | 64.71% | 62.82% | 43.17% | 100.00% | 190 / 29 | 2 |

## Diagnostyka alignmentu

| Wariant | Pierwsze i ostatnie słowo (sekundy) | Śr. score Wav2Vec2* | Score poniżej 0.5 | Liczba mówców | Czas E2E |
|---|---|---:|---:|---:|---:|
| mlx-whisper-large-v3-turbo-pyannote | 0.0 / 182.544 | 0.462 | 149 | 2 | 243.7 s |
| mlx-whisper-large-v3-turbo-whisperx-vad-pyannote-r1 | 22.79 / 173.573 | 0.328 | 238 | 2 | 259.3 s |
| mlx-whisper-large-v3-turbo-whisperx-vad-pyannote-r2 | 22.79 / 173.573 | 0.339 | 235 | 2 | 183.8 s |
| mlx-whisper-large-v3-turbo-whisperx-vad-pyannote-12s | 0.031 / 173.573 | 0.472 | 95 | 2 | 163.6 s |
| mlx-whisper-large-v3-turbo-whisperx-silero-12s-pyannote | 1.026 / 177.054 | 0.463 | 107 | 2 | 189.9 s |

Referencja: 230/247 słów w rozłącznych oknach (93.12%).
*Wynik alignment-score NIE jest miarą dokładności czasów słów.
Kompletne timestampy nie dowodzą trafności ich położenia.
MAE start/end wymaga ludzkiej referencji czasowej, DER wymaga
ręcznie oznaczonego diarization timeline. Oba pozostają niewyznaczone.
E5 i BERTScore liczą osobne workflows na tych samych JSON.
