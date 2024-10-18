import numpy as np
from . import _whisper_cpp

class WhisperModel:
    def __init__(self, model_path, use_gpu=False):
        self.model = _whisper_cpp.WhisperModel(model_path, use_gpu)

    def transcribe(self, audio):
        # Ensure audio is a numpy array of float32
        audio = np.array(audio, dtype=np.float32)
        
        # Run inference
        transcription = self.model.transcribe(audio)
        
        return " ".join(transcription)

    def __del__(self):
        # Explicitly delete the C++ object
        del self.model

def load_model(model_path):
    return WhisperModel(model_path)