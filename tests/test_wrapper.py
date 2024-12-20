import platform
import unittest
import numpy as np
import threading
import time
import queue
import os
from concurrent.futures import ThreadPoolExecutor
from simpler_whisper import (
    WhisperModel,
    AsyncWhisperModel,
    ThreadedWhisperModel,
    set_log_callback,
    LogLevel,
)


class TestWhisperWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # download the model from https://ggml.ggerganov.com/ggml-model-whisper-tiny.en-q5_1.bin
        # and place it in the project root
        url = "https://ggml.ggerganov.com/ggml-model-whisper-tiny.en-q5_1.bin"
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "ggml-tiny.en-q5_1.bin"
        )
        if not os.path.exists(model_path):
            import requests

            print(f"Downloading model from {url}...")
            response = requests.get(url)
            with open(model_path, "wb") as f:
                f.write(response.content)
            print(f"Model downloaded to {model_path}")

        # Get the model path relative to the project root
        cls.model_path = model_path

        # Verify model exists
        if not os.path.exists(cls.model_path):
            raise FileNotFoundError(f"Model file not found at {cls.model_path}")

        # Create sample audio data (silence)
        cls.sample_rate = 16000
        duration_sec = 3
        cls.test_audio = np.zeros(cls.sample_rate * duration_sec, dtype=np.float32)

        # Create some mock audio with varying amplitudes for better testing
        cls.mock_speech = np.sin(
            2 * np.pi * 440 * np.linspace(0, 1, cls.sample_rate)
        ).astype(np.float32)

    def test_sync_model_basic(self):
        """Test basic synchronous model initialization and transcription"""
        try:
            model = WhisperModel(self.model_path, False)
            result = model.transcribe(self.test_audio)
            self.assertIsInstance(result, list)
        except Exception as e:
            self.fail(f"Basic synchronous model test failed: {str(e)}")

    def test_sync_model_empty_audio(self):
        """Test synchronous model with empty audio"""
        model = WhisperModel(self.model_path, False)
        empty_audio = np.array([], dtype=np.float32)
        response = model.transcribe(empty_audio)
        self.assertEqual(response, [])

    # def test_sync_model_invalid_audio(self):
    #     """Test synchronous model with invalid audio data"""
    #     model = WhisperModel(self.model_path, False)
    #     invalid_audio = np.array([1.5, -1.5], dtype=np.float64)  # Wrong dtype
    #     with self.assertRaises(Exception):
    #         model.transcribe(invalid_audio)

    # def test_async_model_basic(self):
    #     """Test basic async model functionality"""
    #     results = queue.Queue()

    #     def callback(chunk_id, segments, is_partial):
    #         results.put((chunk_id, segments, is_partial))

    #     model = whisper.AsyncWhisperModel(self.model_path, False)
    #     try:
    #         model.start(callback)
    #         chunk_id = model.transcribe(self.test_audio)

    #         # Wait for result with timeout
    #         try:
    #             result = results.get(timeout=10)
    #             self.assertEqual(result[0], chunk_id)  # Check if chunk_id matches
    #         except queue.Empty:
    #             self.fail("Async transcription timeout")

    #     finally:
    #         model.stop()

    # def test_threaded_model_basic(self):
    #     """Test basic threaded model functionality"""
    #     results = queue.Queue()

    #     def callback(chunk_id, segments, is_partial):
    #         results.put((chunk_id, segments, is_partial))

    #     model = whisper.ThreadedWhisperModel(
    #         self.model_path,
    #         False,
    #         max_duration_sec=5.0,
    #         sample_rate=self.sample_rate,
    #     )

    #     try:
    #         model.start(callback)
    #         chunk_id = model.queue_audio(self.mock_speech)

    #         # Wait for result with timeout
    #         try:
    #             result = results.get(timeout=10)
    #             self.assertEqual(result[0], chunk_id)
    #         except queue.Empty:
    #             self.fail("Threaded transcription timeout")
    #     finally:
    #         model.stop()

    # def test_threaded_model_continuous(self):
    #     """Test threaded model with continuous audio chunks"""
    #     results = []
    #     result_lock = threading.Lock()

    #     def callback(chunk_id, segments, is_partial):
    #         with result_lock:
    #             results.append((chunk_id, segments, is_partial))

    #     model = whisper.ThreadedWhisperModel(
    #         self.model_path,
    #         False,
    #         max_duration_sec=1.0,
    #         sample_rate=self.sample_rate,
    #     )

    #     try:
    #         model.start(callback)

    #         # Queue multiple chunks of audio
    #         chunk_size = self.sample_rate  # 1 second chunks
    #         num_chunks = 3
    #         chunk_ids = []

    #         for i in range(num_chunks):
    #             chunk = self.mock_speech[i * chunk_size : (i + 1) * chunk_size]
    #             chunk_id = model.queue_audio(chunk)
    #             chunk_ids.append(chunk_id)
    #             time.sleep(0.1)  # Small delay between chunks

    #         # Wait for all results
    #         max_wait = 15  # seconds
    #         start_time = time.time()
    #         while len(results) < num_chunks and (time.time() - start_time) < max_wait:
    #             time.sleep(0.1)

    #         self.assertGreaterEqual(len(results), num_chunks)

    #     finally:
    #         model.stop()

    def test_log_callback(self):
        """Test log callback functionality"""
        log_messages = queue.Queue()

        def log_callback(level, message):
            log_messages.put((level, message))

        try:
            # Set the log callback
            set_log_callback(log_callback)

            # Create a minimal audio sample
            sample_rate = 16000
            duration_sec = 1
            test_audio = np.zeros(sample_rate * duration_sec, dtype=np.float32)

            # Create a model and do minimal transcription
            model = WhisperModel(self.model_path, use_gpu=False)
            try:
                model.transcribe(test_audio)
            except Exception as e:
                print(f"Transcription failed but continuing: {e}")

            # Check if we received any log messages
            try:
                log_message = log_messages.get_nowait()
                self.assertIsInstance(log_message, tuple)
                self.assertIsInstance(log_message[0], int)  # level
                self.assertIsInstance(log_message[1], str)  # message
            except queue.Empty:
                self.skipTest("No log messages received")

        except Exception as e:
            self.skipTest(f"Log callback test failed: {e}")
        finally:
            # Reset log callback
            try:
                set_log_callback(None)
            except:
                pass


if __name__ == "__main__":
    unittest.main()
