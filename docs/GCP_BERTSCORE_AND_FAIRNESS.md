# Google Cloud Speech-to-Text + BERTScore

## Lokalna transkrypcja GCP (macOS Apple Silicon)

1. W Google Cloud Console wybierz lub utwórz projekt z aktywnym
   rozliczaniem i włącz usługę Cloud Speech-to-Text API.
2. Zainstaluj [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).
   Program używa Google **Application Default Credentials (ADC)**.
   Nie wprowadzaj i nie commituj kluczy JSON, tokenów ani API keys.
3. Zainstaluj zależności: uv sync --extra mlx --extra gui --extra mlx-audio --extra gcp.
4. Uruchom: uv run --no-sync python gui.py.
5. Wybierz Silnik ASR = gcp i Model = chirp_3.
   Wpisz Project ID tylko jeśli nie jest skonfigurowany w gcloud/ADC.
   Domyślna lokalizacja przetwarzania to **eu**.
6. Kliknij Połącz z Google Cloud. GUI wywoła gcloud auth
   application-default login, a standardowa autoryzacja otworzy się
   w przeglądarce. Po zakończeniu poświadczenia pozostają w
   systemowym profilu gcloud/ADC, nie w repo ani aplikacji.
7. Uruchom transkrypcję. Program zapyta o zgodę na wysłanie
   audio do GCP i ewentualne naliczanie opłat.

Zamiast przycisku można zalogować się w terminalu:
gcloud auth application-default login
gcloud config set project TWOJ_PROJEKT
gcloud auth application-default set-quota-project TWOJ_PROJEKT

Przykład CLI:
uv run --no-sync python transcribe_mlx.py input/sample-pl.m4a --asr-backend gcp --model chirp_3 --gcp-project TWOJ_PROJEKT --gcp-location eu --diarize-backend mlx-nemotron-8bit

Backend **nie uruchamia API podczas importu**. Audio trafia do Google
dopiero po wybraniu GCP i wykonaniu transkrypcji. Program używa
synchronicznego Speech-to-Text **V2** dla lokalnych WAV ≤12 s,
poniżej limitu 60 sekund / 10 MB na żądanie, oraz autoryzacji ADC.

**Przetwarzanie głosów**: Chirp 3 pl-PL wspiera transkrypcję,
ale natywna google'owska diaryzacja Chirp 3 *nie obejmuje*
języka polskiego. Dlatego po ASR pozostają lokalne: alignment WhisperX
i wybrany Pyannote/Nemotron/Sortformer. Word-level timestamps
powstają przez lokalny alignment; fragment Chirp 3 nie udaje
że ma precyzyjne znaczniki słów przed alignowaniem.

Sygnał audio jest wysyłany do chmury Google: sprawdź podstawę prawną
przetwarzania, poufność, wymagania organizacji i cennik GCP.
Cloud Speech to usługa płatna, niezależna od ChatGPT, Google Drive
czy konta Google. Nie ma automatycznego uploadu do Cloud Storage.

W celu wyłączenia autoryzacji ADC:
gcloud auth application-default revoke

## BERTScore

BERTScore token-level działa w osobnym workflow:
.github/workflows/bertscore-quality.yml

Ręczna referencja: output/sample-pl.primary.reference.md
Warianty ASR: output/hypotheses/
Raport wynikowy: output/bertscore-quality.md

Encoder: bert-base-multilingual-cased; tokenowe kontekstowe
Precision / Recall / F1, osobno dla TK i URZĘDNICZKI SĄDOWEJ,
średnia macro ról, brakującym kwestiom przyznawane są zera.
Dodatkowe alarmy: zmiany numerów lub negacji.
Nie traktuj F1 jako dokładności wyrażonej procentowo, ani jako
zastępstwa WER/CER, podobieństwa zdań E5 czy weryfikacji faktów.

Ręczne uruchomienie:
python -m pip install bert-score==0.3.13 "transformers>=4.48,<5"
python scripts/evaluate_bertscore.py

## Porównanie Parakeet / Whisper z kontrolą diaryzacji

Poprzednia 5-elementowa macierz zmieniała równocześnie ASR
i system diaryzacji. Nie pozwalała osobno przypisać efektu
zmiany quantization Nemotron 8-bit vs BF16.

Nowy eksperyment 2 × 2 znajduje się w:
.github/workflows/fair-asr-matrix.yml

Warunki: to samo input/sample-pl.m4a, identyczne MLX-Silero VAD,
identyczny WhisperX alignment, identyczna rodzina diarizera,
dwa poziomy precyzji Nemotron, dwa silniki ASR.

Parakeet v3 + Nemotron 8-bit
Parakeet v3 + Nemotron BF16
Whisper Turbo + Nemotron 8-bit
Whisper Turbo + Nemotron BF16

Dla każdego z czterech wariantów porównuj wyniki WER/CER i
BERTScore z tą samą ręczną referencją oraz liczbę wykrytych
mówców, ale nie wyciągaj wniosków o DER bez ręcznych czasów
mówców. Jeśli hipotezy z powtórnych przebiegów różnią się,
uwzględnij także wariancję pomiędzy próbami.

GCP nie jest objęte uruchomionym w GitHub porównaniem z prawdziwym
API, dopóki nie zostaną dostarczone ADC lub Workload Identity
i projekt GCP. Testy jednostkowe GCP są całkowicie offline
i nie generują opłat.
