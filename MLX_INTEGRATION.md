# WhisperX MLX Integration

## Overview

Projekt został przebudowany aby wykorzystać MLX na Apple Silicon z zachowaniem zaawansowanej logiki WhisperX (VAD, alignment, diaryzacja).

## Architektura

### Backend Abstractions

Nowa warstwa abstrakcji pozwala na łatwe przełączanie między różnymi silnikami inferencji:

```
whisperx/
├── backends/
│   ├── __init__.py
│   ├── asr_backend.py          # MLX (mlx-whisper) vs CTranslate2 (faster-whisper)
│   ├── vad_backend.py          # MLX Silero vs Pyannote VAD
│   └── diarization_backend.py  # Pyannote (MPS + CPU fallback)
└── pipeline.py                 # Unified pipeline z auto-selekcją backendów
```

### Komponenty Pipeline

| Komponent | Apple Silicon | x86/Linux/Windows |
|-----------|---------------|------------------|
| **ASR** | MLX (mlx-whisper) na MPS/GPU | CTranslate2 (faster-whisper) na CUDA/CPU |
| **VAD** | MLX Silero v6 (planowane) | Pyannote/silero na PyTorch |
| **Alignment** | PyTorch wav2vec2 na CPU | PyTorch wav2vec2 na GPU/CPU |
| **Diarization** | Pyannote na MPS (z CPU fallback) | Pyannote na CUDA/CPU |

### Kluczowe funkcje

1. **Auto-selekcja backendów** - automatycznie wykrywa platformę i wybiera optymalny backend
2. **Auto-quantization** - dla RAM < 12GB automatycznie używa modeli 4-bit
3. **MPS fallback** - jeśli pyannote crashuje na MPS, automatycznie fallback do CPU
4. **Speaker mapping** - mapowanie SPEAKER_XX na prawdziwe imiona
5. **Token-aware prompt** - initial_prompt przycinany do 200 tokenów (tiktoken)

## Użycie

### CLI (nowy skrypt)

```bash
# Podstawowe użycie (auto-detected Apple Silicon)
python transcribe_mlx.py input/nagranie.m4a

# Z określoną liczbą mówców
python transcribe_mlx.py input/nagranie.m4a --speakers 2

# Z limitem mówców
python transcribe_mlx.py input/nagranie.m4a --min-speakers 2 --max-speakers 4

# Wyłączenie diaryzacji
python transcribe_mlx.py input/nagranie.m4a --no-diarization

# Wyłączenie wyrównywania czasu
python transcribe_mlx.py input/nagranie.m4a --no-alignment

# Wymuszenie CPU (zamiast MLX)
python transcribe_mlx.py input/nagranie.m4a --no-mlx

# Wyłączenie MPS dla diaryzacji (zawsze CPU)
python transcribe_mlx.py input/nagranie.m4a --no-mps-diarization

# Wybór modelu
python transcribe_mlx.py input/nagranie.m4a --model large-v3-turbo
```

### Plik konfiguracyjny JSON

Dla każdego pliku audio można stworzyć `<plik>.json` z konfiguracją:

```json
{
  "_instrukcja_dla_ai": "Przeanalizuj dostarczony opis/transkrypt i wygeneruj plik JSON z polami 'initial_prompt' oraz 'speaker_map'. Pamiętaj: 'initial_prompt' to unikalne słowa kluczowe, słownictwo branżowe, imiona i nazwiska osób z nagrania, nazwy własne. Długość 'initial_prompt' MUSI być krótsza niż 224 tokeny.",
  "initial_prompt": "Spotkanie biznesowe, prezentacja projektu, budżet, harmonogram, kamienie milowe, wdrożenie, specyfikacja techniczna, archiwizacja danych.",
  "speaker_map": {
    "SPEAKER_00": "Andrzej Kowalski",
    "SPEAKER_01": "Michał Nowak",
    "SPEAKER_02": "Marcelina Kowalska"
  },
  "num_speakers": 3,
  "min_speakers": 2,
  "max_speakers": 4
}
```

### GUI (PySide6)

```bash
# Instalacja zależności GUI
uv pip install -e ".[gui]"

# Uruchomienie GUI
python gui.py
```

GUI umożliwia:
- Wybór pliku audio
- Ustawienie liczby mówców
- Mapowanie SPEAKER_XX → imiona
- Wprowadzenie kontekstu (initial_prompt)
- Wyświetlanie postępu transkrypcji
- Podgląd wyników

### Python API

```python
from whisperx.pipeline import TranscriptionPipeline

# Automatyczna selekcja backendów
pipeline = TranscriptionPipeline(
    asr_model="large-v3-turbo",
    hf_token="your_hf_token",
    language="pl",
    use_mlx_for_asr=True,  # MLX dla Apple Silicon
    use_mps_for_diarization=True,  # MPS z CPU fallback
    auto_quantize=True,  # Auto-quantization dla niskiego RAM
)

result = pipeline.transcribe(
    "audio.m4a",
    language="pl",
    num_speakers=2,
    initial_prompt="Spotkanie biznesowe, prezentacja...",
    enable_diarization=True,
    enable_alignment=True,
)

pipeline.cleanup()
```

## Instalacja

### Podstawowa (bez MLX)

```bash
uv pip install -e .
```

### Z MLX dla Apple Silicon

```bash
uv pip install -e ".[mlx]"
```

### Z GUI

```bash
uv pip install -e ".[gui]"
```

### Wszystko

```bash
uv pip install -e ".[all]"
```

## Wymagania

### Apple Silicon (M1/M2/M3/M4)

- Python 3.10+
- MLX >= 0.32.0
- mlx-whisper >= 0.4.0
- PyTorch (CPU dla alignment/diarization)
- pyannote-audio >= 4.0.0
- HF_TOKEN w .env

### x86/Linux/Windows

- Python 3.10+
- PyTorch z CUDA (opcjonalnie)
- faster-whisper >= 1.2.0
- pyannote-audio >= 4.0.0
- HF_TOKEN w .env

## Porównanie ze starym skryptem

| Feature | Stary (vendor/transcribe_mlx_pipeline.py) | Nowy (transcribe_mlx.py) |
|---------|-------------------------------------------|--------------------------|
| Backend selekcja | Ręczna (whispermlx) | Automatyczna (MLX/CTranslate2) |
| VAD | Pyannote CPU | MLX Silero (planowane) / Pyannote |
| Diarization | CPU lub MPS (hardcoded) | MPS z CPU fallback |
| Alignment | CPU | CPU (brak MLX portu) |
| Speaker mapping | JSON | JSON + CLI args |
| Initial prompt | Tiktoken clipping | Tiktoken clipping |
| GUI | Brak | PySide6 |
| Formaty wyjścia | SRT, VTT, TSV, MD, JSON | SRT, VTT, TSV, MD, JSON |
| PyTorch MLX support | Nie | Nie (CTranslate2 dla x86) |

## Ograniczenia

1. **MLX VAD** - Silero v6 MLX port istnieje na HuggingFace ale nie jest jeszcze w pełni zintegrowany
2. **Alignment** - wav2vec2 wymaga PyTorch, brak MLX portu
3. **MPS bugs** - pyannote na MPS może crashować na M4 Pro (shader bug), ale mamy auto-fallback do CPU
4. **MLX batched inference** - mlx-whisper nie wspiera batched inference w przeciwieństwie do faster-whisper

## Planowane ulepszenia

1. Pełna integracja MLX Silero VAD
2. Własny MLX embedder dla diaryzacji (ResNet34 + sklearn clustering)
3. MLX port wav2vec2 dla alignment
4. Batched inference w MLX backend
5. Optymalizacja pamięci dla długich nagrań

## Troubleshooting

### MPS crashuje przy diaryzacji

```
Warning: MPS crashed during diarization (shader error), retrying on CPU
```

To znany bug pyannote na M4 Pro. Pipeline automatycznie fallback do CPU.

### Brak MLX na x86

MLX jest tylko dla Apple Silicon. Na x86/Linux/Windows pipeline automatycznie używa CTranslate2.

### Niski RAM

Dla RAM < 12GB pipeline automatycznie używa modeli 4-bit:
- `mlx-community/whisper-large-v3-turbo-q4`
- `mlx-community/whisper-large-v3-mlx-4bit`

Można wyłączyć auto-quantization:
```python
pipeline = TranscriptionPipeline(auto_quantize=False)
```

## Licencja

BSD-2-Clause (taka sama jak upstream WhisperX)
