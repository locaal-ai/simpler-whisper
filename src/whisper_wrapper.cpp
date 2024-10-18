#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <whisper.h>

namespace py = pybind11;

class WhisperModel {
public:
    WhisperModel(const std::string& model_path) {
        ctx = whisper_init_from_file(model_path.c_str());
        if (!ctx) {
            throw std::runtime_error("Failed to initialize whisper context");
        }
    }

    ~WhisperModel() {
        if (ctx) {
            whisper_free(ctx);
        }
    }

    py::list transcribe(py::array_t<float> audio) {
        auto audio_buffer = audio.request();
        float* audio_data = static_cast<float*>(audio_buffer.ptr);
        int n_samples = audio_buffer.size;

        whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        
        if (whisper_full(ctx, params, audio_data, n_samples) != 0) {
            throw std::runtime_error("Whisper inference failed");
        }

        int n_segments = whisper_full_n_segments(ctx);
        py::list transcription;

        for (int i = 0; i < n_segments; i++) {
            const char* text = whisper_full_get_segment_text(ctx, i);
            transcription.append(py::str(text));
        }

        return transcription;
    }

private:
    whisper_context* ctx;
};

PYBIND11_MODULE(_whisper_cpp, m) {
    py::class_<WhisperModel>(m, "WhisperModel")
        .def(py::init<const std::string&>())
        .def("transcribe", &WhisperModel::transcribe);
}