# Jakość transkrypcji polskiej — procedura

## Dane testowe
- W repo znajduje się generator własnej polskiej mowy TTS: scripts/generate_polish_sample.py.
  Po uruchomieniu dodaje 30-sekundowe input/sample_pl_30s.ogg oraz tekst referencyjny.
- W trybie PR testy używają syntetycznych danych i podstawionych modeli.
- W tagowanym release tests/e2e_real.py weryfikuje wszystkie rzeczywiste modele;
  nie jest to dowód dokładności WER/DER dla naturalnych rozmów.

## Metryki
- WER: liczba podstawień, opuszczeń i wstawień słów / liczba słów referencyjnych.
- CER: analogiczna miara znakowa, z normalizacją Unicode.
- DER: oceniany w taktach 0,1 s, z dopasowaniem anonimowych ID mówców.
  Implementacja jest pomocniczym przybliżeniem, nie oficjalnym DER z pyannote.metrics.
- MAE alignmentu: średni bezwzględny błąd początku i końca ręcznie oznaczonych słów.

## Wymagana walidacja na M4
Trzeba wykonać trzy oddzielne rodzaje testów z referencjami:
1. czytelne polskie dialogi (>=2 osoby), 30–180 sekund;
2. pogorszona jakość: telefon, pogłos, szum, ciche słowa;
3. rozmowa nakładająca się: co najmniej dwa głosy jednocześnie.

Dla każdej próbki potrzebne są: audio, referencyjny transkrypt i oznaczone
przedziały speaker_id, a do pomiaru alignmentu również ręczne czasy słów.
Wynik benchmarku powinien osobno prezentować Large-v3 vs Turbo oraz
VAD off vs Silero, bez ukrywania próbek zakończonych błędem.

UWAGA: Automatyczne etykiety mówców i wyniki generowane na syntetycznym głosie
nie są weryfikacją tożsamości ani substytutem ręcznej kontroli nagrania.
