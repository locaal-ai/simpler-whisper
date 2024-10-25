#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <whisper.h>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <iostream>

namespace py = pybind11;

std::string trim(const std::string &str)
{
    size_t start = str.find_first_not_of(" \t\n\r");
    size_t end = str.find_last_not_of(" \t\n\r");

    if (start == std::string::npos) // handles empty string "" and all-whitespace strings like " "
        return "";

    return str.substr(start, end - start + 1);
}

// Global variable to store the Python callback function
py::function g_py_log_callback;

// C++ callback function that will be passed to whisper_log_set
void cpp_log_callback(ggml_log_level level, const char *text, void *)
{
    if (!g_py_log_callback.is_none() && text != nullptr && strlen(text) > 0)
    {
        py::gil_scoped_acquire gil;
        g_py_log_callback(level, std::string(text));
    }
}

// Function to set the log callback
void set_log_callback(py::function callback)
{
    g_py_log_callback = callback;
    whisper_log_set(cpp_log_callback, nullptr);
    ggml_log_set(cpp_log_callback, nullptr);
}

// Original synchronous implementation
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
        params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
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

        std::vector<std::string> segments = transcribe_raw_audio(audio_data, n_samples);

        py::list result;
        for (const auto &segment : segments)
        {
            result.append(segment);
        }

        return result;
    }

    std::vector<std::string> transcribe_raw_audio(const float *audio_data, int n_samples)
    {
        if (whisper_full(ctx, params, audio_data, n_samples) != 0)
        {
            throw std::runtime_error("Whisper inference failed");
        }

        int n_segments = whisper_full_n_segments(ctx);
        std::vector<std::string> transcription;
        for (int i = 0; i < n_segments; i++)
        {
            const char *text = whisper_full_get_segment_text(ctx, i);
            transcription.push_back(std::string(text));
        }

        return transcription;
    }

private:
    whisper_context *ctx;
    whisper_full_params params;
};

struct AudioChunk
{
    std::vector<float> data;
    size_t id;
};

struct TranscriptionResult
{
    size_t chunk_id;
    std::vector<std::string> segments;
    bool is_partial;
};

class ThreadedWhisperModel
{
public:
    ThreadedWhisperModel(const std::string &model_path, bool use_gpu = false,
                         float max_duration_sec = 10.0f, int sample_rate = 16000)
        : running(false), next_chunk_id(0),
          max_samples(static_cast<size_t>(max_duration_sec * sample_rate)),
          current_chunk_id(0), model_path(model_path),
          use_gpu(use_gpu)
    {
    }

    ~ThreadedWhisperModel()
    {
        stop();
    }

    void start(py::function callback, int result_check_interval_ms = 100)
    {
        if (running)
            return;

        running = true;
        result_callback = callback;

        process_thread = std::thread(&ThreadedWhisperModel::processThread, this);
        result_thread = std::thread(&ThreadedWhisperModel::resultThread, this,
                                    result_check_interval_ms);
    }

    void stop()
    {
        if (!running)
            return;
        running = false;

        {
            std::lock_guard<std::mutex> lock(input_mutex);
            input_cv.notify_one();
        }

        {
            std::lock_guard<std::mutex> lock(result_mutex);
            result_cv.notify_one();
        }

        if (process_thread.joinable())
            process_thread.join();
        if (result_thread.joinable())
            result_thread.join();

        // Clear accumulated buffer
        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            accumulated_buffer.clear();
        }
    }

    size_t queueAudio(py::array_t<float> audio)
    {
        auto buffer = audio.request();
        float *data = static_cast<float *>(buffer.ptr);
        size_t n_samples = buffer.size;

        AudioChunk chunk;
        chunk.data.assign(data, data + n_samples);
        chunk.id = next_chunk_id++;

        {
            std::lock_guard<std::mutex> lock(input_mutex);
            input_queue.push(std::move(chunk));
            input_cv.notify_one();
        }

        return chunk.id;
    }

    void setMaxDuration(float max_duration_sec, int sample_rate = 16000)
    {
        max_samples = static_cast<size_t>(max_duration_sec * sample_rate);
    }

private:
    void processAccumulatedAudio(WhisperModel &model, bool force_final = false)
    {
        std::vector<float> process_buffer;
        size_t current_id;

        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            if (accumulated_buffer.empty() || accumulated_buffer.size() < 16000)
                return;

            process_buffer = accumulated_buffer;
            current_id = current_chunk_id;

            // Only clear the buffer if we're processing a final result
            if (force_final || accumulated_buffer.size() >= max_samples)
            {
                accumulated_buffer.clear();
            }
        }

        // Process audio
        std::vector<std::string> segments;
        try
        {
            segments = model.transcribe_raw_audio(process_buffer.data(), process_buffer.size());
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception during transcription: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Unknown exception during transcription" << std::endl;
        }

        if (segments.empty())
        {
            return;
        }

        TranscriptionResult result;
        result.chunk_id = current_id;
        result.segments = segments;
        // Set partial flag based on whether this is a final result
        result.is_partial = !(force_final || process_buffer.size() >= max_samples);

        // Add result to output queue
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            result_queue.push(result);
            result_cv.notify_one();
        }
    }

    void processThread()
    {
        WhisperModel model(this->model_path, this->use_gpu);

        while (running)
        {
            AudioChunk all_chunks;
            bool has_chunk = false;

            // Get next chunk from input queue
            {
                std::unique_lock<std::mutex> lock(input_mutex);
                input_cv.wait(lock, [this]
                              { return !input_queue.empty() || !running; });

                if (!running)
                {
                    // Process any remaining audio as final before shutting down
                    processAccumulatedAudio(model, true);
                    break;
                }

                // take all chunks from the queue and create a single chunk
                while (!input_queue.empty())
                {
                    AudioChunk chunk = std::move(input_queue.front());
                    input_queue.pop();
                    all_chunks.data.insert(all_chunks.data.end(), chunk.data.begin(), chunk.data.end());
                    all_chunks.id = chunk.id;
                    has_chunk = true;
                }
            }

            if (has_chunk)
            {
                // Add to accumulated buffer
                {
                    std::lock_guard<std::mutex> lock(buffer_mutex);
                    size_t old_size = accumulated_buffer.size();
                    accumulated_buffer.resize(old_size + all_chunks.data.size());
                    std::copy(all_chunks.data.begin(), all_chunks.data.end(),
                              accumulated_buffer.begin() + old_size);

                    current_chunk_id = all_chunks.id;
                }

                // Process the accumulated audio
                processAccumulatedAudio(model, false);
            }
        }
    }

    void resultThread(int check_interval_ms)
    {
        while (running)
        {
            std::vector<TranscriptionResult> results;

            {
                std::unique_lock<std::mutex> lock(result_mutex);
                result_cv.wait_for(lock,
                                   std::chrono::milliseconds(check_interval_ms),
                                   [this]
                                   { return !result_queue.empty() || !running; });

                if (!running && result_queue.empty())
                    break;

                while (!result_queue.empty())
                {
                    results.push_back(std::move(result_queue.front()));
                    result_queue.pop();
                }
            }

            if (!results.empty())
            {
                py::gil_scoped_acquire gil;
                for (const auto &result : results)
                {
                    if (result.segments.empty())
                        continue;

                    // concatenate segments into a single string
                    std::string full_text;
                    for (const auto &segment : result.segments)
                    {
                        full_text += segment;
                    }
                    full_text = trim(full_text);
                    if (full_text.empty())
                        continue;

                    if (result_callback)
                    {
                        try
                        {
                            result_callback((int)result.chunk_id, py::str(full_text), result.is_partial);
                        }
                        catch (const std::exception &e)
                        {
                            std::cerr << "Exception in result callback: " << e.what() << std::endl;
                        }
                        catch (...)
                        {
                            std::cerr << "Unknown exception in result callback" << std::endl;
                        }
                    }
                }
            }
        }
    }

    whisper_context *ctx;
    std::atomic<bool> running;
    std::atomic<size_t> next_chunk_id;
    size_t current_chunk_id;

    // Audio accumulation
    std::vector<float> accumulated_buffer;
    size_t max_samples;
    std::mutex buffer_mutex;

    std::thread process_thread;
    std::thread result_thread;

    std::queue<AudioChunk> input_queue;
    std::mutex input_mutex;
    std::condition_variable input_cv;

    std::queue<TranscriptionResult> result_queue;
    std::mutex result_mutex;
    std::condition_variable result_cv;

    py::function result_callback;

    std::string model_path;
    bool use_gpu;
};

PYBIND11_MODULE(_whisper_cpp, m)
{
    // Expose synchronous model
    py::class_<WhisperModel>(m, "WhisperModel")
        .def(py::init<const std::string &, bool>())
        .def("transcribe", &WhisperModel::transcribe);

    py::class_<ThreadedWhisperModel>(m, "ThreadedWhisperModel")
        .def(py::init<const std::string &, bool, float, int>(),
             py::arg("model_path"),
             py::arg("use_gpu") = false,
             py::arg("max_duration_sec") = 10.0f,
             py::arg("sample_rate") = 16000)
        .def("start", &ThreadedWhisperModel::start,
             py::arg("callback"),
             py::arg("result_check_interval_ms") = 100)
        .def("stop", &ThreadedWhisperModel::stop)
        .def("queue_audio", &ThreadedWhisperModel::queueAudio)
        .def("set_max_duration", &ThreadedWhisperModel::setMaxDuration,
             py::arg("max_duration_sec"),
             py::arg("sample_rate") = 16000);

    // Expose logging functionality
    m.def("set_log_callback", &set_log_callback, "Set the log callback function");

    py::enum_<ggml_log_level>(m, "LogLevel")
        .value("NONE", GGML_LOG_LEVEL_NONE)
        .value("INFO", GGML_LOG_LEVEL_INFO)
        .value("WARN", GGML_LOG_LEVEL_WARN)
        .value("ERROR", GGML_LOG_LEVEL_ERROR)
        .value("DEBUG", GGML_LOG_LEVEL_DEBUG)
        .value("CONT", GGML_LOG_LEVEL_CONT)
        .export_values();
}
