# Semantyczna ocena polskiej transkrypcji

Ręczna referencja: output/sample-pl.primary.reference.md
Priorytet: TK i URZĘDNICZKA SĄDOWA, trzeci głos pomijany
jako osobna kwestia referencyjna.

**Cosine embeddingów jest podobieństwem, a nie procentem
poprawności tekstu ani skalibrowanym prawdopodobieństwem.**
Nie można porównywać wartości bezpośrednio między encoderami.
Nie ustalamy arbitralnego progu zaliczenia.

Każda ręcznie zweryfikowana kwestia została sparowana z tekstem
rozpoznanym w jej oknie czasowym. Są to tylko czasy początku
zaokrąglone do sekundy, co ogranicza precyzję pomiaru.
Wypowiedzi poboczne wewnątrz głównych okien mogą nadal wpływać
na wynik; modelowe ID mówców nie są traktowane jako prawda.

## Encoder: intfloat/multilingual-e5-small

| Model ASR / diaryzacja | TK cosine | Urzędniczka cosine | Średnia ról | Alarmy numerów/negacji |
|---|---:|---:|---:|---:|
| mlx-whisper-large-v3-mlx-sortformer-v2 | 0.9187 | 0.9000 | 0.9093 | 15 |
| mlx-audio-parakeet-v3-mlx-nemotron-8bit | 0.9310 | 0.8825 | 0.9068 | 11 |
| mlx-whisper-large-v3-turbo-pyannote | 0.9047 | 0.9022 | 0.9034 | 15 |
| mlx-audio-large-v3-turbo-mlx-nemotron-bf16 | 0.9035 | 0.9032 | 0.9034 | 14 |
| mlx-audio-parakeet-redux-mlx-sortformer-v1 | 0.9140 | 0.8701 | 0.8920 | 14 |

### Dziesięć najsłabiej dopasowanych kwestii najlepszego wariantu

Wariant: mlx-whisper-large-v3-mlx-sortformer-v2

- [00:21] URZĘDNICZKA SĄDOWA | cosine 0.7843 | WER 100.00% | flagi: brak
  - Referencja: Zdążył Pan?
  - ASR: (brak)

- [01:11] TK | cosine 0.8309 | WER 100.00% | flagi: brak
  - Referencja: Karny.
  - ASR: to

- [02:23] TK | cosine 0.8318 | WER 100.00% | flagi: brak
  - Referencja: W sensie?
  - ASR: (brak)

- [00:22] TK | cosine 0.8326 | WER 100.00% | flagi: brak
  - Referencja: Zdążyłem.
  - ASR: (brak)

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

- [02:34] URZĘDNICZKA SĄDOWA | cosine 0.8554 | WER 100.00% | flagi: brak
  - Referencja: Pawilon.
  - ASR: (brak)

- [02:08] URZĘDNICZKA SĄDOWA | cosine 0.8652 | WER 50.00% | flagi: NUMERY
  - Referencja: a pawilon: L. To jest 419.
  - ASR: A pawilon L

---

## Encoder: sdadas/mmlw-e5-small

| Model ASR / diaryzacja | TK cosine | Urzędniczka cosine | Średnia ról | Alarmy numerów/negacji |
|---|---:|---:|---:|---:|
| mlx-whisper-large-v3-mlx-sortformer-v2 | 0.8910 | 0.8744 | 0.8827 | 15 |
| mlx-audio-large-v3-turbo-mlx-nemotron-bf16 | 0.8792 | 0.8823 | 0.8808 | 14 |
| mlx-audio-parakeet-v3-mlx-nemotron-8bit | 0.8987 | 0.8604 | 0.8795 | 11 |
| mlx-whisper-large-v3-turbo-pyannote | 0.8819 | 0.8762 | 0.8791 | 15 |
| mlx-audio-parakeet-redux-mlx-sortformer-v1 | 0.8842 | 0.8532 | 0.8687 | 14 |

### Dziesięć najsłabiej dopasowanych kwestii najlepszego wariantu

Wariant: mlx-whisper-large-v3-mlx-sortformer-v2

- [02:24] URZĘDNICZKA SĄDOWA | cosine 0.8111 | WER 200.00% | flagi: NUMERY
  - Referencja: Z przodu.
  - ASR: W sensie 347 i

- [01:37] URZĘDNICZKA SĄDOWA | cosine 0.8184 | WER 81.82% | flagi: NEGACJA
  - Referencja: Ale ja nie mam obowiązku sprawdzania Panu i szukania, wertowania akt.
  - ASR: do sprawdzania planu i szukamy

- [02:13] URZĘDNICZKA SĄDOWA | cosine 0.8224 | WER 150.00% | flagi: NUMERY
  - Referencja: L-419.
  - ASR: końcówki, co pani

- [02:34] URZĘDNICZKA SĄDOWA | cosine 0.8259 | WER 100.00% | flagi: brak
  - Referencja: Pawilon.
  - ASR: (brak)

- [00:21] URZĘDNICZKA SĄDOWA | cosine 0.8301 | WER 100.00% | flagi: brak
  - Referencja: Zdążył Pan?
  - ASR: (brak)

- [00:22] TK | cosine 0.8364 | WER 100.00% | flagi: brak
  - Referencja: Zdążyłem.
  - ASR: (brak)

- [02:31] TK | cosine 0.8398 | WER 100.00% | flagi: NUMERY
  - Referencja: 13-00.
  - ASR: (brak)

- [02:09] TK | cosine 0.8406 | WER 300.00% | flagi: NUMERY
  - Referencja: To są...?
  - ASR: 1419. To są numery, które powinny już ktoś

- [02:23] TK | cosine 0.8419 | WER 100.00% | flagi: brak
  - Referencja: W sensie?
  - ASR: (brak)

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
