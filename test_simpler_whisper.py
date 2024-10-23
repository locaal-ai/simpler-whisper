import sys

# Remove the current directory from sys.path to avoid conflicts with the installed package
sys.path.pop(0)

import numpy as np
import time
import resampy

from simpler_whisper.whisper import (
    load_model,
    set_log_callback,
    LogLevel,
    ThreadedWhisperModel,
)


def my_log_callback(level, message):
    log_levels = {LogLevel.ERROR: "ERROR", LogLevel.WARN: "WARN", LogLevel.INFO: "INFO"}
    print(f"whisper.cpp [{log_levels.get(level, 'UNKNOWN')}] {message.strip()}")


# Path to your Whisper model file
model_path = R"ggml-tiny.en-q5_1.bin"


def test_simpler_whisper():
    try:
        set_log_callback(my_log_callback)

        # Load the model
        print("Loading the Whisper model...")
        model = load_model(model_path, use_gpu=True)
        print("Model loaded successfully!")

        # Create some dummy audio data
        # In a real scenario, this would be your actual audio data
        print("Creating dummy audio data...")
        dummy_audio = np.random.rand(17000).astype(
            np.float32
        )  # 1 second of random noise at 16kHz
        print("Dummy audio data created.")

        # Run transcription
        print("Running transcription...")
        run_times = []
        for _ in range(10):
            start_time = time.time()
            transcription = model.transcribe(dummy_audio)
            end_time = time.time()
            elapsed_time = end_time - start_time
            run_times.append(elapsed_time)
            print(f"Run {_ + 1}: Transcription took {elapsed_time:.3f} seconds.")

        avg_time = np.mean(run_times)
        min_time = np.min(run_times)
        max_time = np.max(run_times)

        print(f"\nStatistics over 10 runs:")
        print(f"Average time: {avg_time:.3f} seconds")
        print(f"Minimum time: {min_time:.3f} seconds")
        print(f"Maximum time: {max_time:.3f} seconds")

        print("Transcription completed.")
        print("Transcription result:")
        print(transcription)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def test_threaded_whisper():
    def handle_result(chunk_id: int, text: str, is_partial: bool):
        print(f"Chunk {chunk_id} results ({'partial' if is_partial else 'final'}):")
        print(f"  {text}")

    # Create model with 10-second max duration
    model = ThreadedWhisperModel(
        model_path=model_path, use_gpu=True, max_duration_sec=10.0
    )

    # load audio from file with av
    import av
    container = av.open(R"C:\Users\roysh\Downloads\1847363777395929088.mp4")
    audio = container.streams.audio[0]
    print(audio)
    frame_generator = container.decode(audio)

    # Start processing with callback
    print("Starting threaded Whisper model...")
    model.start(callback=handle_result)

    for i, frame in enumerate(frame_generator):
        # print(f"Queueing audio chunk {i + 1}")
        # Read audio chunk
        # resample to 16kHz
        samples = resampy.resample(frame.to_ndarray().mean(axis=0), frame.rate, 16000)

        # Queue some audio (will get partial results until 10 seconds accumulate)
        chunk_id = model.queue_audio(samples)
        # print(f"  Queued chunk {i + 1} with ID {chunk_id} size {len(samples)}")
        # sleep for the size of the audio chunk
        time.sleep(len(samples) / 16000)

    # close the container
    container.close()

    # When done
    print("Stopping threaded Whisper model...")
    model.stop()  # Will process any remaining audio as final


if __name__ == "__main__":
    # test_simpler_whisper()
    test_threaded_whisper()
