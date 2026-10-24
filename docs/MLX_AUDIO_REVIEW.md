# MLX-Audio — ocena integracji z MacWhisper PL

**Analiza kodu upstream:** [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio), commit [76f3593](https://github.com/Blaizzy/mlx-audio/commit/76f35931cfe7db8f1f6b604d7b205ffe9d74ff8d), 24.10.2026; ostatnie opublikowane wydanie v0.6.1 (20.10.2026).

**Stan integracji:** analiza, bez dodawania zależności i bez przełączania działającego MVP. Główna ścieżka MacWhisper pozostaje `mlx-whisper -> polski Wav2Vec2 alignment -> pyannote Community-1 -> eksport`.

## 1. Zakres mlx-audio

[MLX-Audio](https://github.com/Blaizzy/mlx-audio) (kod MIT) jest wielomodelową biblioteką do STT, TTS, STS, VAD/diaryzacji i generacji muzyki na Apple Silicon. Moduły relevantne dla MacWhisper:

- `mlx_audio.stt` — Whisper, Parakeet TDT v3, Nemotron 3.5 streaming, Qwen3-ASR, VibeVoice, i inne;
- `mlx_audio.vad` — MLX Silero, Sortformer, Nemotron 3 Diarization;
- `mlx_audio.stt.eval` — skrypty oceny ASR (normalizacja, WER);
- `mlx_audio.audio_io` — miniaudio z fallbackiem FFmpeg dla formatów M4A/AAC/OGG.

Nie integrować całej biblioteki ani TTS, STS, serwera REST czy web UI tylko po to, by przeprowadzić test backendu.

### Kandydaci do polskiej transkrypcji

| Model | Potwierdzona obsługa PL w dokumentacji upstream | Timestampy | Ocena do testów |
|---|---|---|---|
| Whisper Large-v3 / Turbo | Tak, wielojęzyczny (99+) | Segmenty + słowa (cross-attention/DTW) | Referencja, również konkurent bieżącego `mlx-whisper` |
| Parakeet TDT v3 0.6B | **Tak**, polski na liście 25 języków | Zdania i słowa wg upstream | **Priorytet A**: porównać WER/CER, szybkość i RAM |
| Parakeet Redux | **Tak**, 25 języków jak v3 | Analogiczny wynik TDT; zweryfikować szczegółowo | Priorytet B, ultra-lekkie wagi około 217 MB |
| Nemotron 3.5 ASR 0.6B | Upstream opisuje 40 locale, nie potwierdza jawnie klucza `pl-PL` | Token emission timestamps | Eksperymentalny; sprawdzić `prompt_dictionary` przed testem |
| Qwen3-ASR 0.6/1.7B | **Nie** — oficjalna lista nie zawiera polskiego | Segmenty i osobny forced aligner | Odrzucić jako polski backend MVP |
| VibeVoice ASR Streaming 1.5/7B | **Nie** — lista 10 języków bez PL | Zależne od checkpointu | Odrzucić do ASR po polsku |

Dokumenty źródłowe: [Whisper](https://github.com/Blaizzy/mlx-audio/blob/main/docs/models/stt/whisper.md), [Parakeet](https://github.com/Blaizzy/mlx-audio/blob/main/docs/models/stt/parakeet.md), [Qwen3](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/qwen3_asr/README.md), [Nemotron ASR](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/nemotron_asr/README.md), [VibeVoice](https://github.com/Blaizzy/mlx-audio/blob/main/docs/models/stt/vibevoice-asr.md).

### Diaryzacja i overlapping speakers

**Nemotron 3 Diarization**: [instrukcja i kod](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/nemotron_diarization/README.md). Model MLX zwraca wycinki `(start,end,speaker)` i macierz prawdopodobieństw aktywności `(frames,8)` w interwałach 10 ms; rozpoznaje jednoczesną aktywność kilku głosów. Wersje BF16 i 8-bit. Obsługa offline, długich nagrań przez speaker cache/FIFO i streaming; etykiety tylko lokalne. API:
```python
from mlx_audio.vad import load
diar = load("mlx-community/Nemotron-3-Diarization-8bit", strict=True)
result = diar.generate("audio.wav", threshold=0.5)
segments = [(seg.start, seg.end, seg.speaker) for seg in result.segments]
# result.speaker_probs: frame-by-frame activity, potentially overlapping
```
Ważne: **diaryzacja nie separuje sygnałów audio**. Nawet speaker-masked ASR upstream nie rekonstruuje w pełni słów mówionych jednocześnie. Model ma ograniczenie do maksymalnie ośmiu slotów mówców, a jakość względem polskich rozmów wymaga pomiaru DER, missed speech, false alarms i overlap accuracy.

**Sortformer v1/v2.1**: maksymalnie czterech rozmówców, także streaming. Sensowna alternatywa, ale Nemotron 3 ma większy limit liczby rozmówców. Źródło: [Sortformer](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/sortformer/README.md).

**Silero VAD MLX**: wykrywanie mowy (`predict_proba`, `get_speech_timestamps`) w MLX, bez konieczności ładowania drugiej kopii detektora w Torch. To potencjalny następny etap migracji VAD, po testach zakresów clip timestamps. Źródło: [Silero MLX](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/silero_vad/README.md).

## 2. API i różnice w potoku

Model ASR ładowany przez `from mlx_audio.stt import load`, potem `model.generate(audio)`. Model Whisper ma `language="pl"`, `initial_prompt`, `clip_timestamps`, `word_timestamps=True`, `hotwords`. Zwraca wynik z `.text`, `.segments`, `.language`. Parakeet v3 dostarcza także `.sentences`, ale nie przyjmować bez testów, że wszystkie modele dostarczają identyczne timestampy lub prompt.

Proponowany adapter:
```text
Input -> Decode -> Polish STTBackend -> canonical segments/words
                           |               |
                   mlx-whisper (prod)       +-> Polish forced alignment
                   mlx-audio Whisper (lab)  +-> DiarizationBackend -> speaker attribution
                   Parakeet v3 (lab)        +-> validated export / manifest
```
`DiarizationBackend` powinien normalizować wyniki Pyannote i Nemotron do struktury z osobnymi `start/end/speaker` oraz **listą równolegle aktywnych mówców**. Nie przypisywać na siłę jednego mówcy do słów w overlap. Na razie nie zmieniać produkcyjnych parametrów `initial_prompt` ani pełnej polskiej ścieżki.

## 3. Ryzyka zależności, licencji i pamięci

- W zweryfikowanym `pyproject.toml` upstream `mlx-audio` ma `mlx==0.32.4` oraz `transformers>=5.14.0`. MacWhisper opiera się na WhisperX/Pyannote/Wav2Vec2, PyTorch 2.8 i innych zależnościach; bez testów nie instalować `mlx-audio` na siłę w bazowym środowisku, nie zmieniać `uv.lock` tylko dla eksploracji.
- Przed migracją wykonać osobne środowisko `uv venv .venv-mlx-audio`, przypiąć commit lub stabilne wydanie i testować izolowanie. W przypadku integracji produkcyjnej ponownie rozwiązać zależności i przetestować całą aplikację.
- Licencja kodu `mlx-audio` to MIT, ale **wagi modeli mają własne licencje**. Konwertowane Nemotron-3-Diarization używa warunków upstream OpenMDW 1.1; sprawdzić wymogi redystrybucji przed dołączaniem wagi do release. Modele i wagi powinny być pobierane opcjonalnie, nie bundlowane.
- Obciążenie pamięci należy mierzyć dla całego procesu (MLX active/peak i RSS) z kolejnością modeli i zwalnianiem cache. Natywne MLX dla diaryzacji nie gwarantuje automatycznie niższego łącznego RAM niż Pyannote CPU/MPS.
- Wyniki `mlx-audio` i obecnego `mlx-whisper` mogą mieć inną segmentację; metryki tekstu i diarizacji należy oceniać niezależnie od formatowania.

## 4. Protokół decyzji o migracji

1. Na tym samym zbiorze zweryfikowanych **polskich** nagrań (cicha mowa, rozmowy, szum, co najmniej 2 mówców, overlap) porównać `mlx-whisper` Large-v3/Turbo, `mlx-audio` Whisper i Parakeet v3/Redux: WER, CER, liczby, nazwy własne, RTF, RAM. Oddzielić strumienie od trybu offline.
2. Porównać Pyannote Community-1 vs Nemotron 3 diarization (opcjonalnie Sortformer): DER z optymalnym mapowaniem etykiet, overlap recall, false alarms i stabilność w nagraniach godzinnych. Nie wykorzystywać syntetycznej próbki jednego głosu do rozstrzygania jakości diaryzacji.
3. Porównać czasy słów z obecnego polskiego Wav2Vec2 i natywnego `mlx-audio` Whisper DTW; szczególnie cisza, przyspieszenia i słowa polskie.
4. Dopiero po przekroczeniu ustalonych progów wydajności/jakości wybrać jeden opcjonalny backend do włączenia w GUI/CLI i dodać end-to-end test do GitHub Actions. Domyślna ścieżka pozostaje bez zmian, aż kandydat udowodni przewagę.

**Decyzja na dziś:** priorytetem prototypu MLX Audio jest `Parakeet v3` jako konkurent ASR, a największym możliwym usprawnieniem `Nemotron 3 Diarization` do wielomówczej diaryzacji z overlappem. Nie przywracać nieczynnej opcji `mlx_nemotron` tylko na podstawie dokumentacji — wymaga zaimplementowanego adaptera, testów oraz pomiaru na polskim nagraniu.
