"""Utilities for loading the sentence encoder used by GoAttack."""

import os
from pathlib import Path


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIRECTORY_NAME = "models--sentence-transformers--all-MiniLM-L6-v2"

_CACHED_MODEL = None


def _local_model_directory() -> Path:
    configured_path = os.environ.get("GOATTACK_EMBEDDING_MODEL_DIR")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return Path(__file__).resolve().parent / MODEL_DIRECTORY_NAME


def _is_complete_local_model(model_directory: Path) -> bool:
    has_config = (model_directory / "modules.json").is_file()
    has_weights = any(
        (model_directory / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )
    return has_config and has_weights


def load_embedding_model():
    """Load the local encoder, or download and cache it when absent."""
    global _CACHED_MODEL

    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    model_directory = _local_model_directory()
    has_local_model = _is_complete_local_model(model_directory)
    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
    if not has_local_model and offline:
        raise RuntimeError(
            f"Sentence encoder not found at {model_directory}, but HF_HUB_OFFLINE "
            "is enabled. Disable offline mode and run the model download command "
            "shown in README.md."
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Run: "
            "pip install -r requirements.txt"
        ) from exc

    if has_local_model:
        print(f"Loading sentence encoder from: {model_directory}")
        _CACHED_MODEL = SentenceTransformer(str(model_directory))
        return _CACHED_MODEL

    print(f"Sentence encoder not found locally. Downloading {MODEL_ID} ...")
    try:
        _CACHED_MODEL = SentenceTransformer(MODEL_ID)
        model_directory.mkdir(parents=True, exist_ok=True)
        _CACHED_MODEL.save(str(model_directory))
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {MODEL_ID}. Check the internet connection, or "
            "download it on another machine and copy it to "
            f"{model_directory}. Original error: {exc}"
        ) from exc

    print(f"Sentence encoder cached at: {model_directory}")
    return _CACHED_MODEL
