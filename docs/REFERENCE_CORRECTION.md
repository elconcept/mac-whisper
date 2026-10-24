# Korekta polskiej transkrypcji i budowa referencji

## 1. Tekst: popraw bezpośrednio w GitHub

Otwórz output/sample-pl.reference.md i naciśnij ikonę ołówka / Edit.
Obok lub w drugim oknie odsłuchuj input/sample-pl.m4a.

- Każdy wiersz zaczyna się od znacznika [MM:SS], potem etykiety
  SPEAKER_00/01/02 i wypowiedzi. Poprawiaj wyłącznie na podstawie nagrania.
- Niepewne słowa oznacz [niezrozumiałe], zamiast wymyślać wypowiedź.
- Ujednolić mówców w całym pliku, nie kierując się numeracją modeli.
- Trzeci głos jest odległy i gorzej słyszalny; zbadaj go osobno.
- Wyczyść oznaczenia overlapu dopiero po odsłuchu, nie mechanicznie.
- W nagłówku pliku zmień REFERENCE_STATUS: DRAFT na
  REFERENCE_STATUS: VERIFIED po ukończeniu całej ręcznej korekty.

GitHub Actions wtedy przelicza WER/CER każdej zapisanej hipotezy ASR i zapisuje
output/reference-quality.md. Dopóki plik ma DRAFT, raport nie jest generowany.

## 2. Rozróżnianie rozmówców: opcjonalna referencja DER

W output/sample-pl.speakers.template.json znajduje się **predykcja modelu**.
Nie jest ona referencją. Możesz skopiować ją jako
output/sample-pl.speakers.reference.json, a następnie:

1. Sprawdź każdy interwał start/end z nagraniem (w sekundach).
2. Zmodyfikuj speaker w każdym przedziale tak, aby oznaczał tę samą osobę
   w całym nagraniu; kilka przedziałów z różnymi mówcami może nakładać się.
3. Dodaj brakującą odległą osobę i brakujące albo źle wykryte interwały.
4. Ustaw verified=true **wyłącznie po sprawdzeniu wszystkich przedziałów**.

Następna zmiana w GitHub uruchomi porównanie diaryzacji. Wynik DER w tabeli
jest przybliżeniem liczącym błąd aktywności mówców w interwałach 100 ms;
nie jest to oficjalny DER z collar/overlap protokołu pyannote.metrics.

## 3. Hipotezy modeli i źródła

Oryginalne wyniki maszynowe znajdują się osobno w output/hypotheses/.
Nie modyfikuj ich podczas ręcznej korekty. Każdy JSON zawiera segmenty i
czasy słów, a odpowiedni plik *.report.json informacje o czasie i liczbie
etykiet rozpoznanych rozmówców. Audio źródłowe: input/sample-pl.m4a.

Aktualna kolekcja ma pięć hipotez:
- Parakeet v3 / Nemotron 8-bit
- Parakeet Redux / Sortformer v1
- MLX-Audio Whisper Turbo / Nemotron BF16
- MLX Whisper Large-v3 / Sortformer v2
- MLX Whisper Turbo / Pyannote Community-1

**Dopasowanie liczby etykiet mówców do znanej liczby osób nie oznacza**
poprawnej identyfikacji trzeciej, odległej osoby. Decyzję opieraj na odsłuchu.

## 4. Lokalna kontrola wyników

    python3 scripts/evaluate_reference.py

Tego polecenia nie trzeba uruchamiać po edycji w GitHub, ponieważ workflow
reference-quality.yml wykonuje je automatycznie i wersjonuje raport.
