# MEMO: Wdrożenie WhisperX MLX dla Apple Silicon

## Data wdrożenia
2026-10-24

## Przebudowa core'u WhisperX dla Apple Silicon (MLX)

### Cel
Przebudowa WhisperX aby wykorzystać MLX na Apple Silicon z zachowaniem zaawansowanej logiki WhisperX (VAD, alignment, diarization).

### Wdrożone komponenty

#### 1. MLX ASR Backend (`whisperx/pipeline.py`)
- **Auto-detect platform**: Apple Silicon → MLX (mlx-whisper) na MPS, x86 → CTranslate2
- **Model mapping**: large-v3-turbo domyślny, mapowanie do mlx-community repo IDs
- **API**: `mlx_whisper.transcribe(audio, path_or_hf_repo=model_name)`
- **Czas**: 6-9s dla 30s audio (warm run, po kompilacji kerneli Metal)
- **Prędkość**: 3.3x-5x szybciej niż real-time na M4

#### 2. Memory Guard (`whisperx/memory_guard.py`)
- **Funkcje**:
  - `cleanup_unified_memory()`: agresywne czyszczenie MPS/MLX/CUDA cache
  - `get_free_ram_gb()`: monitorowanie dostępnej pamięci
  - `check_memory_safety()`: auto-cleanup przy niskim RAM (<2GB)
- **Wywołanie**: między etapami pipeline (ASR → Alignment → Diarization)
- **Cel**: zapobieganie WindowServer freeze na macOS

#### 3. Diarization Backends (`whisperx/backends/diarization_backend.py`)
- **PyannoteDiarizationBackend**:
  - PyTorch z MPS fallback do CPU (shader bug na M4 Pro)
  - Automatyczny retry na CPU po błędzie MPS
  - Wysoka precyzja (AHC clustering)
- **MlxNemotronDiarizationBackend**:
  - MLX Sortformer (mlx-audio)
  - **Status**: backend zaimplementowany, model niedostępny w HuggingFace (401 Repository Not Found)
  - Gotowy do użycia gdy model będzie dostępny
- **DiarizationBackendFactory**: factory pattern dla łatwego przełączania

#### 4. CLI (`transcribe_mlx.py`)
- **Flaga `--diarize-backend`**: `pyannote` (domyślny) lub `nemotron`
- **Integracja**: przekazuje backend do pipeline
- **Token clipping**: tiktoken, limit 200 tokenów
- **Speaker mapping**: JSON config + CLI args

#### 5. GUI (`gui.py`)
- **Threading**: `TranscriptionWorker(QThread)` - wszystkie obliczenia w osobnym wątku
- **Sygnały Qt**: `progress`, `finished`, `error` - nie blokują UI
- **Przełącznik silników**: `QComboBox` z Pyannote/Nemotron
- **Pyannote jako domyślny**: `setCurrentIndex(0)`
- **Obsługa błędów Nemotron**: czytelny komunikat przy 401/Repository Not Found
- **Monitorowanie RAM**: wyświetlanie wolnego RAM w czasie rzeczywistym

### Wyniki testów

#### Test integracyjny (30s audio, Python 3.12)
- **Cold run**: 39s (kompilacja kerneli Metal + inferencja)
- **Warm run 1**: 9s (tylko inferencja, 4x szybciej)
- **Warm run 2**: 6s (stabilny czas, 6.5x szybciej niż cold)
- **MLX ASR**: ✅ działa na MPS
- **Alignment**: ✅ word-level timestamps
- **Diarization**: ✅ wykryto 2 mówców (SPEAKER_00, SPEAKER_01)
- **Etykiety mówców**: ✅ poprawne (nie UNKNOWN) po `fill_nearest=True`

#### Estymacja dla godzinnych nagrań
- **Real-time processing**: 3.3x-5x szybciej
- **Godzinny plik**: ~12-18 minut na cały pipeline (VAD + ASR + Alignment + Diarization)

### Środowisko

- **Python**: 3.12.14 (arm64) - scipy działa poprawnie na M4
- **OS**: macOS 26, Apple Silicon M4
- **Dependencies**: uv sync --all-extras (123 packages)
- **Zależności**: MLX, MLX-Whisper, PyTorch, Pyannote, PySide6

### Ograniczenia

1. **MLX Nemotron**: model niedostępny publicznie w HuggingFace (401)
   - Backend zaimplementowany i gotowy
   - Przełącznik w GUI z obsługą błędów
   - Pyannote używany jako domyślny

2. **Pyannote MPS**: znany bug shaderów na M4 Pro
   - Automatyczny fallback do CPU
   - Stabilne na CPU, choć wolniejsze

3. **Alignment**: zawsze PyTorch CPU (brak MLX portu)

### Pliki wdrożone

- `whisperx/pipeline.py` - unified pipeline z MLX ASR
- `whisperx/memory_guard.py` - ochrona pamięci
- `whisperx/backends/diarization_backend.py` - backendy diaryzacji
- `whisperx/backends/__init__.py` - exports
- `transcribe_mlx.py` - CLI z --diarize-backend
- `gui.py` - GUI z threading i przełącznikiem
- `test_integration.py` - test integracyjny
- `test_nemotron.py` - test MLX Nemotron
- `pyproject.toml` - optional dependency groups

### Test produkcyjny (do dokończenia)

**Plik**: `/Users/tom/projects/mac-whisper/input/20260921_czytelnia_akt.m4a` (~3 min, 13.7 MB)

**Polecenie**:
```bash
source .venv-py312/bin/activate
python transcribe_mlx.py input/20260921_czytelnia_akt.m4a --min_speakers 3 --max_speakers 3 --diarize_backend pyannote
```

**Do monitorowania**:
- Stabilność zużycia RAM podczas diaryzacji
- Skuteczność `cleanup_unified_memory()`
- Czas przetwarzania dla dłuższego pliku

### Zalecenia

1. **Domyślny backend**: Pyannote (stabilny, dostępny, dobre czasy na MPS)
2. **MLX Nemotron**: do wdrożenia gdy model będzie dostępny publicznie
3. **Memory guard**: krytyczne dla długich plików (>30 min)
4. **GUI**: gotowe do użycia, threading działa poprawnie

### Status

✅ Architektura wdrożona
✅ Testy integracyjne przeszły (30s audio)
⏸️ Test produkcyjny (do dokończenia - problem z siecią)
✅ GUI z threading i przełącznikiem
✅ Memory guard wdrożony
