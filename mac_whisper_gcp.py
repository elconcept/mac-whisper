"""Google Cloud Speech-to-Text V2 adapter (Polish, EU, Chirp 3).

Uses Application Default Credentials (ADC): existing gcloud login, Workload
Identity or service account from GOOGLE_APPLICATION_CREDENTIALS. No JSON keys,
tokens, browser cookies or credentials are ever stored in this repository.

Audio is sent as short linear-PCM WAV clips by the existing transcription
pipeline. Recognize has hard limits: <60s and <=10MB per request. Billing
and Google's cloud data processing apply when this backend is selected.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

GCP_MODELS = ("chirp_3",)
GCP_DEFAULT_LOCATION = "eu"
GCP_LANGUAGE = "pl-PL"
MAX_INLINE_BYTES = 10_000_000


class GoogleSpeechConfigError(RuntimeError):
    """Actionable cloud setup/auth failure without exposing credentials."""


def find_gcloud():
    """Packaged GUI may have a reduced PATH compared to the macOS terminal."""
    return (shutil.which("gcloud") or
            next((str(p) for p in (
                Path("/opt/homebrew/bin/gcloud"),
                Path("/usr/local/bin/gcloud"),
                Path("/opt/homebrew/share/google-cloud-sdk/bin/gcloud"),
            ) if p.is_file()), None))


def application_default_credentials(project_id=None):
    """Resolve ADC and project. Do not call gcloud login inside a worker."""
    try:
        import google.auth
        credentials, autodetected = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"])
    except Exception as exc:
        raise GoogleSpeechConfigError(
            "Brak autoryzacji Google Cloud. Kliknij 'Połącz z Google Cloud' "
            "w aplikacji (lub uruchom: gcloud auth application-default login). "
            "Po zalogowaniu wybierz projekt z włączonym Cloud Speech-to-Text API."
        ) from exc
    project = (project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
               or os.getenv("GCLOUD_PROJECT") or autodetected)
    if not project:
        gcloud = find_gcloud()
        if gcloud:
            try:
                found = subprocess.run(
                    [gcloud, "config", "get-value", "project"],
                    check=False, capture_output=True, text=True, timeout=5,
                )
                if found.returncode == 0 and found.stdout.strip() not in ("", "(unset)"):
                    project = found.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                pass
    if not project:
        raise GoogleSpeechConfigError(
            "Zalogowano do Google, ale nie znaleziono projektu GCP. "
            "Podaj Project ID w GUI albo --gcp-project; projekt musi mieć "
            "włączone Cloud Speech-to-Text API i rozliczenia."
        )
    if not re.fullmatch(r"[a-z][a-z0-9-]{3,62}", project):
        raise GoogleSpeechConfigError("Nieprawidłowy identyfikator projektu Google Cloud.")
    return credentials, project


def validate_location(location):
    if location not in ("eu", "us"):
        raise ValueError("Cloud Speech V2 Chirp 3 supported deployment: eu or us")
    return location


def gcp_transcribe_clip(path, *, model="chirp_3", project_id=None,
                        location=GCP_DEFAULT_LOCATION, client=None,
                        client_factory=None, credentials=None, duration=None):
    """Call synchronous STT V2 for one WAV clip; return canonical relative times.

    Native Chirp 3 Polish speaker diarization is not available. Use the selected
    Pyannote/Nemotron diarization after local WhisperX forced alignment.
    Word timing is from local alignment, not fabricated from the GCP response.
    """
    if model not in GCP_MODELS:
        raise ValueError("Unknown Google Speech-to-Text model: " + str(model))
    location = validate_location(location)
    data = Path(path).read_bytes()
    if len(data) > MAX_INLINE_BYTES:
        raise ValueError("GCP synchroniczne rozpoznawanie: fragment >10 MB; użyj krótszych fragmentów")
    if duration is not None and duration >= 60:
        raise ValueError("GCP synchroniczne rozpoznawanie: fragment >=60 sekund")
    if credentials is None or not project_id:
        resolved_credentials, resolved_project = application_default_credentials(project_id)
        credentials = credentials or resolved_credentials
        project_id = project_id or resolved_project
    try:
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.types import cloud_speech
        from google.api_core.client_options import ClientOptions
    except ImportError as exc:
        raise GoogleSpeechConfigError(
            "Brakuje integracji GCP. Zainstaluj: uv sync --extra gcp"
        ) from exc
    if client is None:
        factory = client_factory or SpeechClient
        client = factory(credentials=credentials, client_options=ClientOptions(
            api_endpoint=f"{location}-speech.googleapis.com"))
    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[GCP_LANGUAGE], model=model,
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True, enable_word_time_offsets=False),
    )
    req = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{project_id}/locations/{location}/recognizers/_",
        config=config, content=data,
    )
    try:
        response = client.recognize(request=req, timeout=100)
    except Exception as exc:
        # Credentials can be refreshed by google-auth; never log secrets or
        # try to silently switch from EU to the US endpoint.
        raise GoogleSpeechConfigError(
            f"GCP Speech-to-Text ({location}/{model}) nie wykonał transkrypcji. "
            "Sprawdź uprawnienia projektu, quota/billing, włączenie API, "
            "oraz gcloud auth application-default login. "
            f"Typ błędu: {type(exc).__name__}."
        ) from exc
    texts = [
        item.alternatives[0].transcript.strip()
        for item in response.results
        if item.alternatives and item.alternatives[0].transcript.strip()
    ]
    if not texts:
        return client, {"language": "pl", "segments": []}
    # STT Recognize does NOT guarantee utterance timestamps. Keep each
    # clip's original relative bounds, never invent per-phrase times.
    return client, {"language": "pl", "segments": [{
        "start": 0.0, "end": float(duration or 0.0),
        "text": " ".join(texts),
    }]}


def auth_command(project_id=None):
    """An explicit GUI button launches this interactive browser sign-in."""
    gcloud = find_gcloud()
    if not gcloud:
        raise GoogleSpeechConfigError(
            "Nie znaleziono gcloud CLI. Zainstaluj Google Cloud CLI, "
            "a następnie kliknij ponownie 'Połącz z Google Cloud'.")
    cmd = [gcloud, "auth", "application-default", "login"]
    if project_id:
        if not re.fullmatch(r"[a-z][a-z0-9-]{3,62}", project_id):
            raise GoogleSpeechConfigError("Nieprawidłowy Google Cloud Project ID.")
        cmd.extend(["--project", project_id])
    return cmd
