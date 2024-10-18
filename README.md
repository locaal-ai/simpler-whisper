# Simpler Whisper

A simple Python wrapper for whisper.cpp, providing an easy-to-use interface for speech recognition using the Whisper model. This package uses a CMake-based build process to create a Python extension that interfaces with the whisper.cpp library, supporting static libraries on Mac and Linux, and dynamic libraries on Windows.

## Installation

To install simpler-whisper, you need:
- A C++ compiler (e.g., GCC, Clang, or MSVC)
- CMake (version 3.12 or higher)
- NumPy

Then you can install using pip:

```bash
pip install simpler-whisper
```

## Usage

```python
import simpler_whisper.whisper
import numpy as np

# Load the model
model = simpler_whisper.whisper.load_model("path/to/model.bin")

# Load your 16kHz mono audio file as a numpy array of float32
audio = np.frombuffer(open("path/to/audio.raw", "rb").read(), dtype=np.float32)

# Transcribe
transcription = model.transcribe(audio)

print(transcription)
```

## Platform-specific notes

- On Windows, the package uses a DLL (whisper.dll), which is included in the package.
- On Mac and Linux, the package uses a static library (libwhisper.a), which is linked into the extension.

## Building from source

If you're building from source:
1. Clone the repository:
   ```
   git clone https://github.com/locaal-ai/simpler-whisper.git
   cd simpler-whisper
   ```
2. Install the package in editable mode:
   ```
   pip install -e .
   ```

This will run the CMake build process and compile the extension.

## License

This project is licensed under the MIT License - see the LICENSE file for details.