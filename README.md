# MacWhisper PL — polska transkrypcja na Apple Silicon

Lokalna aplikacja do transkrypcji **wyłącznie polskich nagrań** na macOS ARM64
(MLX Whisper), z wymuszonym alignmentem Wav2Vec2 dla języka polskiego,
diaryzacją Pyannote Community-1, eksportem MD/JSON/SRT/VTT/TSV/TXT oraz
manifestem SHA-256 źródłowego pliku audio. Repozytorium rozwija kod
[WhisperX](https://github.com/m-bain/whisperX) (licencja BSD-2-Clause).

## Instalacja

    brew install ffmpeg espeak-ng
    uv sync --extra mlx --extra gui --extra dev

Python 3.12 arm64. Przy pierwszym użyciu zostaną pobrane wagi modeli.
Do użycia Pyannote ustaw HF_TOKEN w środowisku lub lokalnym pliku .env
(po zaakceptowaniu warunków modelu na Hugging Face). ASR bez diaryzacji
nie wymaga tokenu.

## Próbka polska, około 30 sekund

    uv run --no-sync python scripts/generate_polish_sample.py

Tworzy input/sample_pl_30s.ogg i tekst referencyjny. Syntetyczny głos i
tekst są wygenerowane dla testów aplikacji, bez nagrań realnych osób.
Nagranie jest regenerowane; Git nie śledzi folderów z nagraniami.

## Transkrypcja

    uv run --no-sync python transcribe_mlx.py input/sample_pl_30s.ogg --speakers 1
    uv run --no-sync python transcribe_mlx.py --no-diarization --no-alignment
    uv run --no-sync python transcribe_mlx.py nagranie.m4a --model large-v3
    uv run --no-sync python transcribe_mlx.py nagranie.m4a --min-speakers 2 --max-speakers 4
    uv run --no-sync python transcribe_mlx.py nagranie.m4a --vad silero
    uv run --no-sync python gui.py

Model domyślny: large-v3-turbo. Opcja large-v3 służy do porównania jakości.
VAD jest **domyślnie wyłączony** do czasu benchmarku na autentycznych
polskich rozmowach; tryb Silero jest opcjonalny. Silero i MLX nie zastępują
manualnej kontroli wyników, zwłaszcza przy nakładających się głosach.

Opcjonalny sidecar JSON o nazwie nagranie.json:

    {
      "initial_prompt": "Kraków, Kodeks postępowania, EZVIZ",
      "num_speakers": 2,
      "speaker_map": {"SPEAKER_00": "Osoba A"}
    }

Alternatywnie podaj min_speakers i max_speakers (nie razem z num_speakers).
Jawne opcje CLI/GUI mają pierwszeństwo przed sidecarem, a potem ustawieniami
domyślnymi. Tekst promptu jest limitowany rzeczywistym tokenizerem Whisper.

## Wyniki

Dla nagrania.m4a otrzymasz w output/ transkrypcję w wybranym formacie i
nagranie.manifest.json z sumą SHA-256 źródła, wersją parametrów, informacją
o środowisku, odczytami pamięci i etapami potoku. Nie modyfikujemy źródłowego
nagrania. Etykiety SPEAKER_00 itd. są estymacją modelu, **nie dowodem
tożsamości rozmówcy**.

## Testy, release i jakość

Zobacz [CI i build macOS](docs/CI_RELEASE.md) oraz
[pomiar jakości polskiej mowy](docs/QUALITY_PL.md).

GitHub Actions uruchamia testy na ARM64 macOS, a dla tagów v* wymaga
prawdziwego przebiegu E2E (MLX + alignment + Pyannote), zanim zbuduje
i opublikuje GitHub Release. Pakiet .app jest podpisany ad-hoc; bez
notaryzacji Apple może wymagać ręcznego pozwolenia na uruchomienie.

Nie commituj autentycznych nagrań, dokumentów i transkrypcji do repozytorium.
