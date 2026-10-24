# Benchmark polskiej mowy — główny dialog TK ↔ URZĘDNICZKA SĄDOWA

Główna referencja: output/sample-pl.primary.reference.md (VERIFIED, ręczna korekta)
Pomocnicza wcześniejsza transkrypcja pełna: output/sample-pl.reference.md
Source audio: input/sample-pl.m4a
Priorytet: TK = SPEAKER_00, URZĘDNICZKA SĄDOWA = SPEAKER_01.
SPEAKER_02 i kwestie poza ręcznie wskazanym głównym dialogiem są poboczne.
Pełne hipotezy modeli i rozmowy z trzecią osobą są zachowane.

**Ważne:** pomocnicza pełna referencja pochodzi z wcześniejszej korekty
i nie została ponownie zsynchronizowana z najnowszymi poprawkami.
Jej WER jest jedynie historycznym punktem odniesienia, NIE główną oceną.

UWAGA: oznaczenia w referencji mają dokładność tylko 1 sekundy
i brak czasów końca. Wyniki 00/01 są **przybliżonym WER/CER
w oknach czasowych**, a nie prawdziwym diarized WER.
Modelowe speaker ID nie służą do wybierania słów w oknach.
Zachowane są więc pomyłki w rozpoznaniu mowy i zakłócenia wewnątrz
głównych wypowiedzi; omijane są odrębne okna SPEAKER_02.
Pokrycie głównej referencji: 93.1%
(230 / 247 słów).
Pominięte sekundy ze zbieżnymi głosami: [122, 162, 172].

| Model | WER TK/urzędniczka (macro) | WER TK | WER urzędniczka | WER dialogu łącznie | CER dialogu | WER dawnej pełnej ref. | DER approx. |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlx-audio-large-v3-turbo-mlx-nemotron-bf16 | 44.42% | 43.75% | 45.10% | 35.22% | 24.36% | 39.80% | — |
| mlx-audio-parakeet-v3-mlx-nemotron-8bit | 47.76% | 40.62% | 54.90% | 38.70% | 26.83% | 46.71% | — |
| mlx-whisper-large-v3-turbo-pyannote | 49.32% | 47.66% | 50.98% | 35.22% | 23.86% | 45.72% | — |
| mlx-whisper-large-v3-mlx-sortformer-v2 | 50.51% | 42.19% | 58.82% | 40.87% | 26.73% | 45.07% | — |
| mlx-audio-parakeet-redux-mlx-sortformer-v1 | 54.91% | 46.09% | 63.73% | 47.83% | 36.14% | 53.29% | — |

### Pomocnicza metryka wobec starszej pełnej referencji

| Model | WER dawnej pełnej ref. | CER dawnej pełnej ref. |
|---|---:|---:|
| mlx-audio-large-v3-turbo-mlx-nemotron-bf16 | 39.80% | 29.22% |
| mlx-audio-parakeet-v3-mlx-nemotron-8bit | 46.71% | 33.13% |
| mlx-whisper-large-v3-turbo-pyannote | 45.72% | 35.92% |
| mlx-whisper-large-v3-mlx-sortformer-v2 | 45.07% | 33.58% |
| mlx-audio-parakeet-redux-mlx-sortformer-v1 | 53.29% | 39.16% |

Wyniki dialogu oceniają wyłącznie ręcznie wybrane kwestie.
Czasy początku zaokrąglono do sekund; słowa przypisano do
rozłącznych okien, a nie według automatycznych speaker ID.
Speaker DER is **not calculated** until an independently manually
verified output/sample-pl.speakers.reference.json provides start/end/overlap
intervals. If supplied, the table contains a 100-ms-frame approximate
speaker-time error after speaker-label permutation, NOT pyannote's
official collar- and overlap-adjusted DER. This approximation is sensitive
to exact boundaries and should not be treated as legal evidence.
Model predictions alone do not establish individual identities.
