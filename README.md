# Simpler Whisper

![Build and Test](https://img.shields.io/github/actions/workflow/status/locaal-ai/simpler-whisper/build.yaml)

A zero-dependency simple Python wrapper for [whisper.cpp](https://github.com/ggerganov/whisper.cpp), providing an easy-to-use interface for speech recognition using the Whisper model. 

Why is it better than [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [pywhispercpp](https://github.com/abdeladim-s/pywhispercpp):
- Zero-dependency: Everything is shipped with the built wheel, no Python dependency (on `av` or `ctranslate2` etc.) except for `numpy`.
- Dead simple API: call `.transcribe()` and get a result
- Acceleration enabled: supports whatever whisper.cpp supports
- Updated: using precompiled whisper.cpp from https://github.com/locaal-ai/occ-ai-dep-whispercpp
- Build time: builds in 2 minutes because it's using a precompiled binary

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

# Load the model file. 
# It's on you to download one from https://huggingface.co/ggerganov/whisper.cpp
model = simpler_whisper.whisper.load_model("path/to/model.bin")

# Load your 16kHz mono audio samples as a numpy array of float32
# It's on you if you need to convert (use av) or resample (use resampy)
audio = np.frombuffer(open("path/to/audio.raw", "rb").read(), dtype=np.float32)

# Transcribe
transcription = model.transcribe(audio)

print(transcription)
```

## Platform-specific notes

- On Windows, the package uses a DLL (whisper.dll), which is included in the package.
- On Mac and Linux, the package uses static libraries that are linked into the extension.

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

### Example: Building for Windows with CUDA acceleration

```powershell
$env:SIMPLER_WHISPER_ACCELERATION=cuda
pip install .
```

### Example: Building for macOS ARM64

```bash
SIMPLER_WHISPER_PLATFORM=arm64 pip install .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.