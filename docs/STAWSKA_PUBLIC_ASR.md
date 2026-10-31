# Stawska — publiczna, wielomodelowa transkrypcja

Właściciel nagrania wyraził zgodę na publikację audio w **publicznym** repozytorium.
Rozmowa zawiera dane osób trzecich. Materiał i wynikowe transkrypcje będą dostępne
publicznie, a usunięcie pliku z repozytorium nie oznacza cofnięcia wcześniejszych kopii.

Nagranie oryginalne: `AUDIO_20260719_2257_TK_Krystyna_Stawska.m4a`, ~345 MB.
**Nie zastępuj źródła wersją po transkodowaniu, jeśli ma zachować metadane EXIF.**

## Zalecana metoda dostarczenia audio (bez Git LFS)

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
