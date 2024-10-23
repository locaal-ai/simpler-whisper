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

// Original synchronous implementation
class WhisperModel
{
public:
    WhisperModel(const std::string &model_path, bool use_gpu = false)
    {
        whisper_context_params ctx_params = whisper_context_default_params();
        ctx_params.use_gpu = use_gpu;
        std::cout << "WhisperModel c'tor Loading model from path: " << model_path << std::endl;
        ctx = whisper_init_from_file_with_params(model_path.c_str(), ctx_params);
        if (!ctx)
        {
            std::cout << "Failed to initialize whisper context" << std::endl;
            throw std::runtime_error("Failed to initialize whisper context");
        }
    }

    ~WhisperModel()
    {
        if (ctx)
        {
            std::cout << "WhisperModel d'tor Freeing whisper context" << std::endl;
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
        std::cout << "Transcribing audio with " << n_samples << " samples" << std::endl;
        std::cout << "first sample: " << audio_data[0] << std::endl;
        std::cout << "last sample: " << audio_data[n_samples - 1] << std::endl;

        whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        if (whisper_full(ctx, params, audio_data, n_samples) != 0)
        {
            std::cout << "Whisper inference failed" << std::endl;
            throw std::runtime_error("Whisper inference failed");
        }
        std::cout << "Whisper inference succeeded" << std::endl;

        int n_segments = whisper_full_n_segments(ctx);
        std::vector<std::string> transcription;
        for (int i = 0; i < n_segments; i++)
        {
            const char *text = whisper_full_get_segment_text(ctx, i);
            transcription.push_back(std::string(text));
        }
        std::cout << "num segments: " << n_segments << std::endl;

        return transcription;
    }

private:
    whisper_context *ctx;
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
          accumulated_samples(0), current_chunk_id(0), model_path(model_path),
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
            accumulated_samples = 0;
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
            std::cout << "Processing accumulated audio with size: " << accumulated_buffer.size() << std::endl;
            if (accumulated_buffer.empty())
                return;

            // check if buffer has less than 1 second of audio
            if (accumulated_samples < 16000)
            {
                std::cout << "Not enough audio to process" << std::endl;
                return;
            }

            process_buffer = accumulated_buffer;
            current_id = current_chunk_id;

            // Only clear the buffer if we're processing a final result
            if (force_final || accumulated_samples >= max_samples)
            {
                accumulated_buffer.clear();
                accumulated_samples = 0;
            }
        }

        // Process audio
        std::cout << "Processing audio with size: " << process_buffer.size() << std::endl;
        std::cout << "Pointer to first sample: " << process_buffer.data() << std::endl;
        std::cout << "First sample: " << process_buffer[0] << std::endl;
        std::vector<std::string> segments = model.transcribe_raw_audio(process_buffer.data(), process_buffer.size());

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
        std::cout << "Starting process thread." << std::endl;

        std::cout << "Loading model from path: " << this->model_path << std::endl;
        WhisperModel model(this->model_path, this->use_gpu);
        std::cout << "Model loaded." << std::endl;

        while (running)
        {
            AudioChunk chunk;
            bool has_chunk = false;

            // Get next chunk from input queue
            {
                std::unique_lock<std::mutex> lock(input_mutex);
                input_cv.wait(lock, [this]
                              { return !input_queue.empty() || !running; });

                if (!running)
                {
                    // Process any remaining audio as final before shutting down
                    std::cout << "Shutting down, processing remaining audio as final." << std::endl;
                    processAccumulatedAudio(model, true);
                    break;
                }

                if (!input_queue.empty())
                {
                    chunk = std::move(input_queue.front());
                    input_queue.pop();
                    has_chunk = true;
                    std::cout << "Got chunk with ID: " << chunk.id << " and size: " << chunk.data.size() << std::endl;
                }
            }

            if (has_chunk)
            {
                // Add to accumulated buffer
                {
                    std::lock_guard<std::mutex> lock(buffer_mutex);
                    size_t old_size = accumulated_buffer.size();
                    accumulated_buffer.resize(old_size + chunk.data.size());
                    std::copy(chunk.data.begin(), chunk.data.end(),
                              accumulated_buffer.begin() + old_size);

                    accumulated_samples += chunk.data.size();
                    current_chunk_id = chunk.id;
                    std::cout << "Accumulated buffer size: " << accumulated_buffer.size() << std::endl;
                }

                // Process the accumulated audio
                std::cout << "Processing accumulated audio." << std::endl;
                processAccumulatedAudio(model, false);
            }
        }
        std::cout << "Exiting process thread." << std::endl;
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
                std::cout << "Got " << results.size() << " results." << std::endl;
                py::gil_scoped_acquire gil;
                for (const auto &result : results)
                {
                    // concatenate segments into a single string
                    std::string full_text;
                    for (const auto &segment : result.segments)
                    {
                        full_text += segment;
                    }
                    std::cout << "Calling result callback with ID: " << result.chunk_id << std::endl;
                    if (result_callback)
                    {
                        result_callback(result.chunk_id, full_text, result.is_partial);
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
    size_t accumulated_samples;
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
        .value("ERROR", GGML_LOG_LEVEL_ERROR)
        .value("WARN", GGML_LOG_LEVEL_WARN)
        .value("INFO", GGML_LOG_LEVEL_INFO)
        .export_values();
}
