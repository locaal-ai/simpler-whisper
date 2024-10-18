import numpy as np

import simpler_whisper.whisper

def test_simpler_whisper():
    # Path to your Whisper model file
    # Replace this with the path to your actual model file
    model_path = R"ggml-model-whisper-tiny.en.bin"

    try:
        # Load the model
        print("Loading the Whisper model...")
        model = simpler_whisper.whisper.load_model(model_path)
        print("Model loaded successfully!")

        # Create some dummy audio data
        # In a real scenario, this would be your actual audio data
        print("Creating dummy audio data...")
        dummy_audio = np.random.rand(17000).astype(np.float32)  # 1 second of random noise at 16kHz
        print("Dummy audio data created.")

        # Run transcription
        print("Running transcription...")
        transcription = model.transcribe(dummy_audio)
        
        print("Transcription completed.")
        print("Transcription result:")
        print(transcription)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_simpler_whisper()
