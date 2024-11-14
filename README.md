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

There are three ways to use simpler-whisper:

### 1. Basic Usage
```python
from simpler_whisper.whisper import WhisperModel

# Load the model (models can be downloaded from https://huggingface.co/ggerganov/whisper.cpp)
model = WhisperModel("path/to/model.bin", use_gpu=True)

# Load and prepare your audio
# You can use av, librosa, or any method that gives you 16kHz mono float32 samples
import av
container = av.open("audio.mp3")
audio_stream = container.streams.audio[0]
samples = np.concatenate([
    frame.to_ndarray().mean(axis=0) if frame.format.channels == 2 else frame.to_ndarray()
    for frame in container.decode(audio_stream)
])

# Transcribe
transcription = model.transcribe(samples)
for segment in transcription:
    print(f"{segment.text} ({segment.t0:.2f}s - {segment.t1:.2f}s)")
```

### 2. Async Processing

This will create a thread in the backend (not locked by the GIL) to allow for asynchronous transcription.

```python
from simpler_whisper.whisper import AsyncWhisperModel

def handle_result(chunk_id: int, segments: List[WhisperSegment], is_partial: bool):
    text = " ".join([seg.text for seg in segments])
    print(f"Chunk {chunk_id}: {text}")

# Create and start async model
model = AsyncWhisperModel("path/to/model.bin", callback=handle_result, use_gpu=True)
model.start()

# Queue audio chunks for processing
chunk_id = model.transcribe(audio_samples)

# When done
model.stop()
```

### 3. Real-time Threaded Processing

This method creates a background thread for real-time transcription that will continuously
process the input in e.g. 10 seconds chunks and report on both final or partial results.

```python
from simpler_whisper.whisper import ThreadedWhisperModel

def handle_result(chunk_id: int, segments: List[WhisperSegment], is_partial: bool):
    text = " ".join([seg.text for seg in segments])
    print(f"Chunk {chunk_id}: {text}")

# Create and start threaded model with 10-second chunks
model = ThreadedWhisperModel(
    "path/to/model.bin",
    callback=handle_result,
    use_gpu=True,
    max_duration_sec=10.0
)
model.start()

# Queue audio chunks as they arrive
chunk_id = model.queue_audio(audio_samples)

# When done
model.stop()
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

### Example: Building for Windows with CUDA acceleration

```powershell
$env:SIMPLER_WHISPER_ACCELERATION="cuda"
pip install .
```

### Example: Building for macOS ARM64

```bash
pip install .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
