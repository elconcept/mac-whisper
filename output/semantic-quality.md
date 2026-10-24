# Semantyczna ocena polskiej transkrypcji

Ręczna referencja: output/sample-pl.primary.reference.md
Priorytet: TK i URZĘDNICZKA SĄDOWA, trzeci głos pomijany
jako osobna kwestia referencyjna.

**Cosine embeddingów jest podobieństwem, a nie procentem
poprawności tekstu ani skalibrowanym prawdopodobieństwem.**
Nie można porównywać wartości bezpośrednio między encoderami.
Nie ustalamy arbitralnego progu zaliczenia.
Pusta hipoteza ASR dostaje specjalną karę 0.0 (BRAK TRANSKRYPCJI),
a nie cosine wektora pustego tekstu, który byłby myląco wysoki.

Każda ręcznie zweryfikowana kwestia została sparowana z tekstem
rozpoznanym w jej oknie czasowym. Są to tylko czasy początku
zaokrąglone do sekundy, co ogranicza precyzję pomiaru.
Wypowiedzi poboczne wewnątrz głównych okien mogą nadal wpływać
na wynik; modelowe ID mówców nie są traktowane jako prawda.

## Encoder: intfloat/multilingual-e5-small

| Model ASR / diaryzacja | TK cosine | Urzędniczka cosine | Średnia ról | Puste kwestie | Alarmy |
|---|---:|---:|---:|---:|---:|
| mlx-whisper-large-v3-mlx-sortformer-v2 | 0.7980 | 0.8089 | 0.8035 | 5 | 19 |
| mlx-audio-parakeet-v3-mlx-nemotron-8bit | 0.8104 | 0.7457 | 0.7781 | 6 | 16 |
| mlx-whisper-large-v3-turbo-pyannote | 0.7457 | 0.7658 | 0.7558 | 7 | 20 |
| mlx-whisper-large-v3-turbo-mlx-sortformer-v2 | 0.7440 | 0.7658 | 0.7549 | 7 | 20 |
| mlx-audio-large-v3-turbo-mlx-sortformer-v2 | 0.6649 | 0.8121 | 0.7385 | 8 | 20 |
| mlx-audio-large-v3-turbo-mlx-nemotron-bf16 | 0.6630 | 0.8121 | 0.7376 | 8 | 20 |
| mlx-whisper-large-v3-turbo-whisperx-vad-pyannote-r1 | 0.7173 | 0.6655 | 0.6914 | 8 | 24 |
| mlx-whisper-large-v3-turbo-whisperx-vad-pyannote-r2 | 0.6742 | 0.6665 | 0.6703 | 9 | 26 |
| mlx-audio-parakeet-redux-mlx-sortformer-v1 | 0.7125 | 0.5026 | 0.6075 | 13 | 24 |

### Dziesięć najsłabiej dopasowanych kwestii najlepszego wariantu

Wariant: mlx-whisper-large-v3-mlx-sortformer-v2

- [00:21] URZĘDNICZKA SĄDOWA | cosine 0.0000 | WER 100.00% | flagi: BRAK TRANSKRYPCJI
  - Referencja: Zdążył Pan?
  - ASR: (brak)

- [00:22] TK | cosine 0.0000 | WER 100.00% | flagi: BRAK TRANSKRYPCJI
  - Referencja: Zdążyłem.
  - ASR: (brak)

- [02:23] TK | cosine 0.0000 | WER 100.00% | flagi: BRAK TRANSKRYPCJI
  - Referencja: W sensie?
  - ASR: (brak)

- [02:31] TK | cosine 0.0000 | WER 100.00% | flagi: BRAK TRANSKRYPCJI, NUMERY
  - Referencja: 13-00.
  - ASR: (brak)

- [02:34] URZĘDNICZKA SĄDOWA | cosine 0.0000 | WER 100.00% | flagi: BRAK TRANSKRYPCJI
  - Referencja: Pawilon.
  - ASR: (brak)

- [01:11] TK | cosine 0.8309 | WER 100.00% | flagi: brak
  - Referencja: Karny.
  - ASR: to

- [02:09] TK | cosine 0.8350 | WER 300.00% | flagi: NUMERY
  - Referencja: To są...?
  - ASR: 1419. To są numery, które powinny już ktoś

- [02:13] URZĘDNICZKA SĄDOWA | cosine 0.8407 | WER 150.00% | flagi: NUMERY
  - Referencja: L-419.
  - ASR: końcówki, co pani

- [02:10] URZĘDNICZKA SĄDOWA | cosine 0.8437 | WER 85.71% | flagi: brak
  - Referencja: To są numery, które powinien ktoś odebrać.
  - ASR: odebrać.

- [02:24] URZĘDNICZKA SĄDOWA | cosine 0.8531 | WER 200.00% | flagi: NUMERY
  - Referencja: Z przodu.
  - ASR: W sensie 347 i

---

## Encoder: sdadas/mmlw-e5-small

| Model ASR / diaryzacja | TK cosine | Urzędniczka cosine | Średnia ról | Puste kwestie | Alarmy |
|---|---:|---:|---:|---:|---:|
| mlx-whisper-large-v3-mlx-sortformer-v2 | 0.7711 | 0.7824 | 0.7768 | 5 | 19 |
| mlx-audio-parakeet-v3-mlx-nemotron-8bit | 0.7788 | 0.7226 | 0.7507 | 6 | 16 |
| mlx-whisper-large-v3-turbo-pyannote | 0.7225 | 0.7377 | 0.7301 | 7 | 20 |
| mlx-whisper-large-v3-turbo-mlx-sortformer-v2 | 0.7194 | 0.7377 | 0.7286 | 7 | 20 |
| mlx-audio-large-v3-turbo-mlx-sortformer-v2 | 0.6425 | 0.7903 | 0.7164 | 8 | 20 |
| mlx-audio-large-v3-turbo-mlx-nemotron-bf16 | 0.6399 | 0.7903 | 0.7151 | 8 | 20 |
| mlx-whisper-large-v3-turbo-whisperx-vad-pyannote-r1 | 0.7028 | 0.6461 | 0.6744 | 8 | 24 |
| mlx-whisper-large-v3-turbo-whisperx-vad-pyannote-r2 | 0.6631 | 0.6476 | 0.6554 | 9 | 26 |
| mlx-audio-parakeet-redux-mlx-sortformer-v1 | 0.6848 | 0.4860 | 0.5854 | 13 | 24 |

### Dziesięć najsłabiej dopasowanych kwestii najlepszego wariantu

Wariant: mlx-whisper-large-v3-mlx-sortformer-v2

- [00:21] URZĘDNICZKA SĄDOWA | cosine 0.0000 | WER 100.00% | flagi: BRAK TRANSKRYPCJI
  - Referencja: Zdążył Pan?
  - ASR: (brak)

- [00:22] TK | cosine 0.0000 | WER 100.00% | flagi: BRAK TRANSKRYPCJI
  - Referencja: Zdążyłem.
  - ASR: (brak)

- [02:23] TK | cosine 0.0000 | WER 100.00% | flagi: BRAK TRANSKRYPCJI
  - Referencja: W sensie?
  - ASR: (brak)

- [02:31] TK | cosine 0.0000 | WER 100.00% | flagi: BRAK TRANSKRYPCJI, NUMERY
  - Referencja: 13-00.
  - ASR: (brak)

- [02:34] URZĘDNICZKA SĄDOWA | cosine 0.0000 | WER 100.00% | flagi: BRAK TRANSKRYPCJI
  - Referencja: Pawilon.
  - ASR: (brak)

- [02:24] URZĘDNICZKA SĄDOWA | cosine 0.8111 | WER 200.00% | flagi: NUMERY
  - Referencja: Z przodu.
  - ASR: W sensie 347 i

- [01:37] URZĘDNICZKA SĄDOWA | cosine 0.8184 | WER 81.82% | flagi: NEGACJA
  - Referencja: Ale ja nie mam obowiązku sprawdzania Panu i szukania, wertowania akt.
  - ASR: do sprawdzania planu i szukamy

- [02:13] URZĘDNICZKA SĄDOWA | cosine 0.8224 | WER 150.00% | flagi: NUMERY
  - Referencja: L-419.
  - ASR: końcówki, co pani

- [02:09] TK | cosine 0.8406 | WER 300.00% | flagi: NUMERY
  - Referencja: To są...?
  - ASR: 1419. To są numery, które powinny już ktoś

- [02:10] URZĘDNICZKA SĄDOWA | cosine 0.8457 | WER 85.71% | flagi: brak
  - Referencja: To są numery, które powinien ktoś odebrać.
  - ASR: odebrać.

---

## Ważne ograniczenia

Negacja i numery (np. 73-39 lub L-419) są dodatkowo flagowane,
ponieważ duży cosine nie gwarantuje niezmienionego znaczenia.
Flagi są sygnałami do sprawdzenia, a nie rozstrzygnięciem.
WER/CER z dokładnym dopasowaniem leksykalnym pozostają w
output/reference-quality.md; DER wymaga odrębnej referencji
z granicami czasowymi każdego głosu.
