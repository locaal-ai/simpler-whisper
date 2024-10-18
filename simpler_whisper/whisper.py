import numpy as np
from . import _whisper_cpp


class WhisperModel:
    def __init__(self, model_path: str, use_gpu=False):
        self.model = _whisper_cpp.WhisperModel(model_path, use_gpu)

    def transcribe(self, audio):
        # Ensure audio is a numpy array of float32
        audio = np.array(audio, dtype=np.float32)

        # Run inference
        transcription = self.model.transcribe(audio)

        return " ".join(transcription)

    def __del__(self):
        # Explicitly delete the C++ object
        if hasattr(self, "model"):
            del self.model


def load_model(model_path: str, use_gpu=False) -> WhisperModel:
    return WhisperModel(model_path, use_gpu)


def set_log_callback(callback):
    """
    Set a custom logging callback function.

    The callback function should accept two arguments:
    - level: An integer representing the log level (use LogLevel enum for interpretation)
    - message: A string containing the log message

    Example:
    def my_log_callback(level, message):
        print(f"[{LogLevel(level).name}] {message}")

    set_log_callback(my_log_callback)
    """
    _whisper_cpp.set_log_callback(callback)


# Expose LogLevel enum from C++ module
LogLevel = _whisper_cpp.LogLevel
