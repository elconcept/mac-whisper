# BERTScore — polski główny dialog

Ręczna referencja: output/sample-pl.primary.reference.md
Model tokenowy: bert-base-multilingual-cased
Zakres: TK i URZĘDNICZKA SĄDOWA. Trzeci rozmówca nie jest oceniany.

BERTScore P/R/F1 jest nieliniowym podobieństwem kontekstowych
wektorów tokenów, NIE procentem dosłownie poprawnej transkrypcji.
Brak rozpoznanej kwestii otrzymuje P=R=F1=0.
Ocena uwzględnia osobno alarmy numerów i negacji.
Czasy źródłowe są zaokrąglone do sekund i okna są przybliżone.

| Model ASR + diaryzacja | F1 TK | F1 urzędniczki | F1 macro | Precision macro | Recall macro | Braki | Alarmy |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlx-audio-parakeet-v3-mlx-nemotron-8bit | 0.7369 | 0.6257 | 0.6813 | 0.6846 | 0.6798 | 6 | 11 |
| mlx-whisper-large-v3-mlx-sortformer-v2 | 0.7004 | 0.6550 | 0.6777 | 0.6819 | 0.6752 | 5 | 15 |
| mlx-audio-large-v3-turbo-mlx-nemotron-bf16 | 0.5954 | 0.7054 | 0.6504 | 0.6547 | 0.6472 | 8 | 14 |
| mlx-whisper-large-v3-turbo-pyannote | 0.6438 | 0.6566 | 0.6502 | 0.6550 | 0.6471 | 7 | 15 |
| mlx-audio-large-v3-turbo-mlx-sortformer-v2 | 0.5949 | 0.7054 | 0.6501 | 0.6543 | 0.6471 | 8 | 14 |
| mlx-whisper-large-v3-turbo-mlx-sortformer-v2 | 0.6388 | 0.6566 | 0.6477 | 0.6530 | 0.6442 | 7 | 15 |
| mlx-audio-parakeet-redux-mlx-sortformer-v1 | 0.6392 | 0.4297 | 0.5344 | 0.5344 | 0.5359 | 13 | 14 |

## Dziesięć najsłabszych fragmentów najlepszego wariantu

Model: mlx-audio-parakeet-v3-mlx-nemotron-8bit

- [00:21] URZĘDNICZKA SĄDOWA: F1 0.0000
  - Referencja: Zdążył Pan?
  - ASR: (brak)
  - Alerty: brak

- [00:22] TK: F1 0.0000
  - Referencja: Zdążyłem.
  - ASR: (brak)
  - Alerty: brak

- [01:06] URZĘDNICZKA SĄDOWA: F1 0.0000
  - Referencja: Na wydział, jaki to będzie, ale to będzie który?
  - ASR: (brak)
  - Alerty: brak

- [02:23] TK: F1 0.0000
  - Referencja: W sensie?
  - ASR: (brak)
  - Alerty: brak

- [02:31] TK: F1 0.0000
  - Referencja: 13-00.
  - ASR: (brak)
  - Alerty: NUMERY

- [02:34] URZĘDNICZKA SĄDOWA: F1 0.0000
  - Referencja: Pawilon.
  - ASR: (brak)
  - Alerty: brak

- [02:13] URZĘDNICZKA SĄDOWA: F1 0.5856
  - Referencja: L-419.
  - ASR: końcówki, co Pani
  - Alerty: NUMERY

- [02:01] URZĘDNICZKA SĄDOWA: F1 0.6215
  - Referencja: 73-39.
  - ASR: 49, 73, 39,
  - Alerty: NUMERY

- [02:08] URZĘDNICZKA SĄDOWA: F1 0.6224
  - Referencja: a pawilon: L. To jest 419.
  - ASR: To
  - Alerty: NUMERY

- [02:09] TK: F1 0.6260
  - Referencja: To są...?
  - ASR: czterta dziewiętnaście, to są numery, które tu
  - Alerty: brak

WER/CER oraz sentence embeddings pozostają osobnymi
metrykami. Wysoki BERTScore nie dowodzi poprawności liczb i zaprzeczeń.
