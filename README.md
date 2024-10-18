# Simpler Whisper

![Build and Test](https://github.com/locaal-ai/simpler-whisper/workflows/Build%20and%20Test/badge.svg)

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

## Build Configuration

Simpler Whisper supports various build configurations to optimize for different hardware and acceleration methods. You can specify the build configuration using environment variables:

- `SIMPLER_WHISPER_ACCELERATION`: Specifies the acceleration method. Options are:
  - `cpu` (default)
  - `cuda` (for NVIDIA GPUs)
  - `hipblas` (for AMD GPUs)
  - `vulkan` (for cross-platform GPU acceleration)

- `SIMPLER_WHISPER_PLATFORM`: Specifies the target platform. This is mainly used for macOS builds to differentiate between x86_64 and arm64 architectures.

### Example: Building with CUDA acceleration

```bash
SIMPLER_WHISPER_ACCELERATION=cuda pip install simpler-whisper
```

### Example: Building for macOS ARM64

```bash
SIMPLER_WHISPER_PLATFORM=arm64 pip install simpler-whisper
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.