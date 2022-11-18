"""Settings file."""
import os
from typing import Any, Dict


def get_value(key: str) -> Any:
    """Returns a value from the corresponding env var or from settings if env var doesn't exist."""
    settings: Dict[str, Any] = {
        # Trained ML/AI model settings
        "TRAINED_MODEL_MODULE_NAME": "sample_model",
        # Module settings
        "MULTITHREADED_INIT": True,
    }
    return os.environ[key] if key in os.environ else settings.get(key, False)
