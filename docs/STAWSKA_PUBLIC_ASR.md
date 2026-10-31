# Stawska — publiczna, wielomodelowa transkrypcja

Właściciel nagrania wyraził zgodę na publikację audio w **publicznym** repozytorium.
Rozmowa zawiera dane osób trzecich. Materiał i wynikowe transkrypcje będą dostępne
publicznie, a usunięcie pliku z repozytorium nie oznacza cofnięcia wcześniejszych kopii.

Nagranie oryginalne: `AUDIO_20260719_2257_TK_Krystyna_Stawska.m4a`, ~345 MB.
**Nie zastępuj źródła wersją po transkodowaniu, jeśli ma zachować metadane EXIF.**

## Podział oryginalnego nagrania na części poniżej 100 MiB

GitHub ogranicza do 100 MiB pojedynczy plik w zwykłej historii Git;
**limit nie dotyczy plików w GitHub Releases** (pojedynczy asset może mieć do 2 GiB).
Dla zachowania preferowanego podziału źródło dzielimy **bajt po bajcie**, bez FFmpeg,
bez dekodowania, transkodowania i zmian metadanych oryginału.

Na Macu w katalogu repozytorium uruchom:

```bash
python3 scripts/split_stawska_audio.py \
  "/ścieżka/do/AUDIO_20260719_2257_TK_Krystyna_Stawska.m4a" \
  --output-dir input/stawska-release --part-mib 75
```

Wynik: `AUDIO_20260719_2257_TK_Krystyna_Stawska.m4a.part-000`,
`...part-001`, kolejne części (dla 345 MB będzie ich pięć) oraz
`AUDIO_20260719_2257_TK_Krystyna_Stawska.m4a.sha256` i kontrolny JSON.
Żaden plik części nie przekroczy 75 MiB; cały oryginał pozostaje niezmieniony.

### Publikacja w Releases i automatyczne uruchomienie Actions

```bash
gh release create stawska-20260719 \
  --repo elconcept/mac-whisper --title "Stawska ASR input" \
  --notes "Owner-authorized multipart recording for ASR." --draft

gh release upload stawska-20260719 \
  input/stawska-release/AUDIO_20260719_2257_TK_Krystyna_Stawska.m4a.part-* \
  input/stawska-release/AUDIO_20260719_2257_TK_Krystyna_Stawska.m4a.sha256 \
  --repo elconcept/mac-whisper

gh release edit stawska-20260719 --repo elconcept/mac-whisper --draft=false
```

Uwaga: assety należy przesłać **przed opublikowaniem** release, bo zdarzenie
`release: published` uruchamia workflow. W przypadku istniejącego już
opublikowanego release użyj ręcznie `workflow_dispatch` ze źródłem `release`.

Workflow pobiera wszystkie części, sprawdza numerację od `part-000`
bez luk, skleja je operacją `cat` i **weryfikuje SHA-256 oryginalnego M4A**
przed uruchomieniem jakiejkolwiek transkrypcji. Brak części albo niezgodność
sumy przerywają zadanie. W Git nie są przechowywane gigabajty audio.

## Zalecana metoda dostarczenia audio w całości (alternatywnie, bez Git LFS)

1. GitHub → elconcept/mac-whisper → Releases → Draft a new release.
2. Tag `stawska-20260719`, dodaj audio jako asset o dokładnej nazwie:
   `AUDIO_20260719_2257_TK_Krystyna_Stawska.m4a`.
3. Publish release. Workflow `Stawska - three Polish ASR hypotheses` powinien
   uruchomić trzy przebiegi (Whisper Turbo, Whisper large-v3, Parakeet v3).
4. Wyniki: Actions → run → Artifacts (7 dni).
5. W workflow_dispatch wybierz `scope=pilot`, aby uruchomić tylko pierwsze 10 min,
   albo `scope=full` dla całości.

Asset jest pobierany na runner do `input/` na czas zadania.
Oryginał nie jest dodawany do historii Git i nie jest modyfikowany.

## Alternatywa — Git LFS

Jeżeli zamiast release wybierasz `input/`, skonfiguruj Git LFS:
```sh
git lfs install
git lfs track "input/AUDIO_20260719_2257_TK_Krystyna_Stawska.m4a"
git add .gitattributes
git add -f input/AUDIO_20260719_2257_TK_Krystyna_Stawska.m4a
git commit -m "data: add owner-authorized Stawska audio through LFS"
git push origin main
```
Dostęp Git LFS i transfer mogą podlegać odrębnemu limitowi GitHub.

## Przebieg i ograniczenia

Mac runner `macos-15` jest ARM64 M1. Publiczne standardowe Actions nie
zużywają prywatnego przydziału minut. Zadania dzielą nagranie na
10-minutowe porcje PCM tylko w środowisku roboczym, a każdy model
generuje odrębne JSON. `combined.json` i `combined.md` zachowują czas
względem początku oryginału.

Nie wykonywana jest diarizacja ani przypisywanie TK/KS na podstawie ASR.
Nie należy automatycznie zastępować transkrypcji po korekcie ludzkiej
hipotezą jednego modelu; konieczne jest porównanie z faktycznym odsłuchem.
