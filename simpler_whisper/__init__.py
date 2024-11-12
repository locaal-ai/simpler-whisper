try:
    from . import _whisper_cpp
    __all__ = ['_whisper_cpp']
except ImportError as e:
    import sys
    print(f"Error importing _whisper_cpp: {e}", file=sys.stderr)
    raise
