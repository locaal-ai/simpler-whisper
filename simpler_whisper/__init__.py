try:
    from ._whisper_cpp import *  # Import everything from _whisper_cpp first
    from .whisper import (  # Then import our wrapper classes
        WhisperModel,
        AsyncWhisperModel,
        ThreadedWhisperModel,
        WhisperSegment,
        WhisperToken,
        set_log_callback,
        LogLevel,
    )

    __all__ = [
        "WhisperModel",
        "AsyncWhisperModel",
        "ThreadedWhisperModel",
        "WhisperSegment",
        "WhisperToken",
        "set_log_callback",
        "LogLevel",
    ]
except ImportError as e:
    import sys

    print(f"Error importing modules: {e}", file=sys.stderr)
    raise
