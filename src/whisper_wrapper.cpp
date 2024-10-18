#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <whisper.h>

namespace py = pybind11;

class WhisperModel
{
public:
    WhisperModel(const std::string &model_path, bool use_gpu = false)
    {
        whisper_context_params ctx_params = whisper_context_default_params();
        ctx_params.use_gpu = use_gpu;
        ctx = whisper_init_from_file_with_params(model_path.c_str(), ctx_params);
        if (!ctx)
        {
            throw std::runtime_error("Failed to initialize whisper context");
        }
    }

    ~WhisperModel()
    {
        if (ctx)
        {
            whisper_free(ctx);
        }
    }

    py::list transcribe(py::array_t<float> audio)
    {
        auto audio_buffer = audio.request();
        float *audio_data = static_cast<float *>(audio_buffer.ptr);
        int n_samples = audio_buffer.size;

        whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

        if (whisper_full(ctx, params, audio_data, n_samples) != 0)
        {
            throw std::runtime_error("Whisper inference failed");
        }

        int n_segments = whisper_full_n_segments(ctx);
        py::list transcription;

        for (int i = 0; i < n_segments; i++)
        {
            const char *text = whisper_full_get_segment_text(ctx, i);
            transcription.append(py::str(text));
        }

        return transcription;
    }

private:
    whisper_context *ctx;
};

// Global variable to store the Python callback function
py::function g_py_log_callback;

// C++ callback function that will be passed to whisper_log_set
void cpp_log_callback(ggml_log_level level, const char *text, void *)
{
    if (!g_py_log_callback.is_none())
    {
        g_py_log_callback(level, text);
    }
}

// Function to set the log callback
void set_log_callback(py::function callback)
{
    g_py_log_callback = callback;
    whisper_log_set(cpp_log_callback, nullptr);
    ggml_log_set(cpp_log_callback, nullptr);
}

PYBIND11_MODULE(_whisper_cpp, m)
{
    py::class_<WhisperModel>(m, "WhisperModel")
        .def(py::init<const std::string &, bool>())
        .def("transcribe", &WhisperModel::transcribe);

    m.def("set_log_callback", &set_log_callback, "Set the log callback function");

    py::enum_<ggml_log_level>(m, "LogLevel")
        .value("ERROR", GGML_LOG_LEVEL_ERROR)
        .value("WARN", GGML_LOG_LEVEL_WARN)
        .value("INFO", GGML_LOG_LEVEL_INFO)
        .export_values();
}