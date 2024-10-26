import av
import argparse
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


log_levels = {LogLevel.ERROR: "ERROR", LogLevel.WARN: "WARN", LogLevel.INFO: "INFO"}


def my_log_callback(level, message):
    if message is not None and len(message.strip()) > 0:
        print(f"whisper.cpp [{log_levels.get(level, 'UNKNOWN')}] {message.strip()}")


# Path to your Whisper model file
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Test simpler-whisper model.")
parser.add_argument("model_path", type=str, help="Path to the Whisper model file")
parser.add_argument("audio_file", type=str, help="Path to the audio file")
# non-positoinal required arg for the method to use (regular vs threaded)
parser.add_argument(
    "method",
    type=str,
    choices=["regular", "threaded"],
    help="The method to use for testing the model",
)
args = parser.parse_args()

model_path = args.model_path
audio_file = args.audio_file


def get_samples_from_frame(frame: av.AudioFrame) -> np.ndarray:
    """
    Extracts and processes audio samples from an audio frame.
    This function reads an audio chunk from the provided audio frame, converts it to mono if it is stereo,
    normalizes the audio if it is in int16 format, and resamples it to 16kHz if necessary.
    Parameters:
    frame (av.AudioFrame): The input audio frame containing the audio data.
    Returns:
    numpy.ndarray: The processed audio samples, normalized and resampled to 16kHz if needed.
    """
    # Read audio chunk
    incoming_audio = frame.to_ndarray()
    # check if stereo
    if incoming_audio.shape[0] == 2:
        incoming_audio = incoming_audio.mean(axis=0)
    # check if the type is int16 or float32
    if incoming_audio.dtype == np.int16:
        incoming_audio = incoming_audio / 32768.0  # normalize to [-1, 1]
    # resample to 16kHz if needed
    if frame.rate != 16000:
        samples = resampy.resample(incoming_audio, frame.rate, 16000)
    else:
        samples = incoming_audio

    return samples


def test_simpler_whisper():
    set_log_callback(my_log_callback)

    # Load the model
    print("Loading the Whisper model...")
    model = load_model(model_path, use_gpu=True)
    print("Model loaded successfully!")

    # Load audio from file with av
    container = av.open(audio_file)
    audio = container.streams.audio[0]
    print(audio)

    frame_generator = container.decode(audio)

    # Run transcription
    print("Running transcription...")
    run_times = []
    samples_for_transcription = np.array([])
    for i, frame in enumerate(frame_generator):
        samples = get_samples_from_frame(frame)
        # append the samples to the samples_for_transcription
        samples_for_transcription = np.append(samples_for_transcription, samples)

        # if there are less than 30 seconds of audio, append the samples and continue to the next frame
        if len(samples_for_transcription) < 16000 * 30:
            continue

        start_time = time.time()
        transcription = model.transcribe(samples_for_transcription)
        end_time = time.time()
        elapsed_time = end_time - start_time
        run_times.append(elapsed_time)
        print(f"Run {i + 1}: Transcription took {elapsed_time:.3f} seconds.")
        for segment in transcription:
            for j, tok in enumerate(segment.tokens):
                print(f"Token {j}: {tok.text} ({tok.t0:.3f} - {tok.t1:.3f})")
        # reset the samples_for_transcription
        samples_for_transcription = np.array([])

    avg_time = np.mean(run_times)
    min_time = np.min(run_times)
    max_time = np.max(run_times)

    print(f"\nStatistics over runs:")
    print(f"Average time: {avg_time:.3f} seconds")
    print(f"Minimum time: {min_time:.3f} seconds")
    print(f"Maximum time: {max_time:.3f} seconds")

    print("Transcription completed.")


def test_threaded_whisper():
    set_log_callback(my_log_callback)

    def handle_result(chunk_id: int, text: str, is_partial: bool):
        print(
            f"Chunk {chunk_id} results ({'partial' if is_partial else 'final'}): {text}"
        )

    # Create model with 10-second max duration
    model = ThreadedWhisperModel(
        model_path=model_path,
        callback=handle_result,
        use_gpu=True,
        max_duration_sec=10.0,
    )

    # load audio from file with av
    container = av.open(audio_file)
    audio = container.streams.audio[0]
    print(audio)
    frame_generator = container.decode(audio)

    # Start processing with callback
    print("Starting threaded Whisper model...")
    model.start()

    for i, frame in enumerate(frame_generator):
        try:
            samples = get_samples_from_frame(frame)

            # Queue some audio (will get partial results until 10 seconds accumulate)
            chunk_id = model.queue_audio(samples)
            # sleep for the size of the audio chunk
            time.sleep(float(len(samples)) / float(16000))
        except:
            break

    # close the container
    container.close()

    # When done
    print("Stopping threaded Whisper model...")
    model.stop()  # Will process any remaining audio as final


if __name__ == "__main__":
    if args.method == "regular":
        test_simpler_whisper()
    else:
        test_threaded_whisper()
