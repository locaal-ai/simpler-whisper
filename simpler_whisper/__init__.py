try:
    from ._whisper_cpp import *
except ImportError as e:
    import sys
    print(f"Error importing modules: {e}", file=sys.stderr)
try:
    from .whisper import (
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
    __all__ = []
