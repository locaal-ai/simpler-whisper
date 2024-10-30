import numpy as np
from typing import Callable, List, Union
from . import _whisper_cpp
from dataclasses import dataclass


@dataclass
class WhisperToken:
    """A token from the Whisper model with timing and probability information."""

    id: int
    p: float
    t0: int  # Start time in milliseconds
    t1: int  # End time in milliseconds
    text: str


@dataclass
class WhisperSegment:
    """A segment of transcribed text with timing information and token details."""

    text: str
    start: int  # Start time in milliseconds
    end: int  # End time in milliseconds
    tokens: List[WhisperToken]


class WhisperModel:
    def __init__(self, model_path: str, use_gpu=False):
        self.model = _whisper_cpp.WhisperModel(model_path, use_gpu)

    def transcribe(self, audio: Union[np.ndarray, List[float]]) -> List[WhisperSegment]:
        # Ensure audio is a numpy array of float32
        audio = np.array(audio, dtype=np.float32)

        # Run inference
        transcription = self.model.transcribe(audio)

        return transcription

    def __del__(self):
        # Explicitly delete the C++ object
        if hasattr(self, "model"):
            del self.model


class AsyncWhisperModel:
    """
    AsyncWhisperModel is a class that provides asynchronous transcription of audio data using a Whisper model.
    """

    def __init__(
        self,
        model_path: str,
        callback: Callable[[int, List[WhisperSegment], bool], None],
        use_gpu=False,
    ):
        self.model = _whisper_cpp.AsyncWhisperModel(model_path, use_gpu)
        self._is_running = False
        self.callback = callback

    def transcribe(self, audio: Union[np.ndarray, List[float]]) -> int:
        """
        Transcribes the given audio input using the model.
        Args:
            audio (Union[np.ndarray, List[float]]): The audio data to be transcribed.
                It can be either a numpy array or a list of floats.
        Returns:
            int: The queued chunk ID.
        """
        # Ensure audio is a numpy array of float32
        audio = np.array(audio, dtype=np.float32)

        # Run async inference (no return value)
        return self.model.transcribe(audio)

    def handle_result(
        self, chunk_id: int, segments: List[WhisperSegment], is_partial: bool
    ):
        if self.callback is not None:
            self.callback(chunk_id, segments, is_partial)

    def start(self, result_check_interval_ms=100):
        """
        Start the processing threads with a callback for results.

        Args:
            callback: Function that takes three arguments:
                     - chunk_id (int): Unique identifier for the audio chunk
                     - segments (WhisperSegment): Transcribed text for the audio chunk
                     - is_partial (bool): Whether this is a partial result
            result_check_interval_ms (int): How often to check for results
        """
        if self._is_running:
            return

        self.model.start(self.handle_result, result_check_interval_ms)
        self._is_running = True

    def stop(self):
        """
        Stop processing and clean up resources.
        Any remaining audio will be processed as a final segment.
        """
        if not self._is_running:
            return

        self.model.stop()
        self._is_running = False

    def __del__(self):
        # Explicitly delete the C++ object
        if hasattr(self, "model"):
            if self._is_running:
                self.stop()
                self._is_running = False
            del self.model


class ThreadedWhisperModel:
    def __init__(
        self,
        model_path: str,
        callback: Callable[[int, List[WhisperSegment], bool], None],
        use_gpu=False,
        max_duration_sec=10.0,
        sample_rate=16000,
    ):
        """
        Initialize a threaded Whisper model for continuous audio processing.

        Args:
            model_path (str): Path to the Whisper model file
            use_gpu (bool): Whether to use GPU acceleration
            max_duration_sec (float): Maximum duration in seconds before finalizing a segment
            sample_rate (int): Audio sample rate (default: 16000)
            callback: Function that takes three arguments:
                     - chunk_id (int): Unique identifier for the audio chunk
                     - segments (List[WhisperSegment]): Transcribed text for the audio chunk
                     - is_partial (bool): Whether this is a partial result
        """
        self.model = _whisper_cpp.ThreadedWhisperModel(
            model_path, use_gpu, max_duration_sec, sample_rate
        )
        self._is_running = False
        self.callback = callback

    def handle_result(
        self, chunk_id: int, segments: List[WhisperSegment], is_partial: bool
    ):
        if self.callback is not None:
            self.callback(chunk_id, segments, is_partial)

    def start(self, result_check_interval_ms=100):
        """
        Start the processing threads with a callback for results.

        Args:
            callback: Function that takes three arguments:
                     - chunk_id (int): Unique identifier for the audio chunk
                     - segments (WhisperSegment): Transcribed text for the audio chunk
                     - is_partial (bool): Whether this is a partial result
            result_check_interval_ms (int): How often to check for results
        """
        if self._is_running:
            return

        self.model.start(self.handle_result, result_check_interval_ms)
        self._is_running = True

    def stop(self):
        """
        Stop processing and clean up resources.
        Any remaining audio will be processed as a final segment.
        """
        if not self._is_running:
            return

        self.model.stop()
        self._is_running = False

    def queue_audio(self, audio):
        """
        Queue audio for processing.

        Args:
            audio: Audio samples as numpy array or array-like object.
                  Will be converted to float32.

        Returns:
            chunk_id (int): Unique identifier for this audio chunk
        """
        # Ensure audio is a numpy array of float32
        audio = np.array(audio, dtype=np.float32)
        return self.model.queue_audio(audio)

    def set_max_duration(self, max_duration_sec, sample_rate=16000):
        """
        Change the maximum duration for partial segments.

        Args:
            max_duration_sec (float): New maximum duration in seconds
            sample_rate (int): Audio sample rate (default: 16000)
        """
        self.model.set_max_duration(max_duration_sec, sample_rate)

    def __del__(self):
        # Ensure threads are stopped and resources cleaned up
        if hasattr(self, "model"):
            if self._is_running:
                self.stop()
            del self.model


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
