#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

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

struct WhisperToken
{
    int id;
    float p;
    int64_t t0;
    int64_t t1;
    std::string text;
};

struct WhisperSegment
{
    std::string text;
    int64_t start;
    int64_t end;
    std::vector<WhisperToken> tokens;
};

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
        params.no_timestamps = false;
        params.token_timestamps = true;
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
        py::list result;
        // Check if input is empty
        if (audio.is_none() || audio.size() == 0)
        {
            return result;
        }

        auto audio_buffer = audio.request();
        float *audio_data = static_cast<float *>(audio_buffer.ptr);
        int n_samples = audio_buffer.size;

        std::vector<WhisperSegment> segments = transcribe_raw_audio(audio_data, n_samples);

        for (const auto &segment : segments)
        {
            result.append(py::cast(segment));
        }

        return result;
    }

    std::vector<WhisperSegment> transcribe_raw_audio(const float *audio_data, int n_samples)
    {
        if (whisper_full(ctx, params, audio_data, n_samples) != 0)
        {
            throw std::runtime_error("Whisper inference failed");
        }

        int n_segments = whisper_full_n_segments(ctx);
        std::vector<WhisperSegment> transcription;
        for (int i = 0; i < n_segments; i++)
        {
            const char *text = whisper_full_get_segment_text(ctx, i);
            WhisperSegment segment;
            segment.start = whisper_full_get_segment_t0(ctx, i);
            segment.end = whisper_full_get_segment_t1(ctx, i);
            segment.text = std::string(text);
            const int n_tokens = whisper_full_n_tokens(ctx, i);
            for (int j = 0; j < n_tokens; ++j)
            {
                // get token
                whisper_token_data token =
                    whisper_full_get_token_data(ctx, i, j);
                WhisperToken wt;
                wt.id = token.id;
                wt.p = token.p;
                wt.t0 = token.t0;
                wt.t1 = token.t1;
                wt.text = std::string(whisper_token_to_str(ctx, token.id));
                segment.tokens.push_back(wt);
            }

            transcription.push_back(segment);
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
    bool is_partial;
    std::vector<WhisperSegment> segments;
};

class AsyncWhisperModel
{
public:
    AsyncWhisperModel(const std::string &model_path, bool use_gpu = false) : model_path(model_path), use_gpu(use_gpu),
                                                                             running(false), next_chunk_id(0), current_chunk_id(0)
    {
    }

    ~AsyncWhisperModel()
    {
    }

    void start(py::function callback, int result_check_interval_ms = 100)
    {
        if (running)
            return;

        running = true;
        result_callback = callback;

        process_thread = std::thread(&AsyncWhisperModel::processThread, this);
        result_thread = std::thread(&AsyncWhisperModel::resultThread, this,
                                    result_check_interval_ms);
    }

    /**
     * @brief Transcribes the given audio data.
     *
     * This function takes an audio input in the form of a py::array_t<float> and
     * processes it by queuing the audio for transcription.
     *
     * @param audio A py::array_t<float> containing the audio data to be transcribed.
     * @return size_t The queued chunk ID.
     */
    size_t transcribe(py::array_t<float> audio)
    {
        // Check if input is empty
        if (audio.is_none() || audio.size() == 0)
        {
            return 0;
        }

        return this->queueAudio(audio);
    }

    virtual void stop()
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

protected:
    virtual void processThread()
    {
        WhisperModel model(model_path, use_gpu);

        while (running)
        {
            AudioChunk chunk;
            // Get next chunk from input queue
            {
                std::unique_lock<std::mutex> lock(input_mutex);
                input_cv.wait_for(lock,
                                  std::chrono::milliseconds(100),
                                  [this]
                                  { return !input_queue.empty() || !running; });

                if (!running)
                    break;

                if (input_queue.empty())
                    continue;

                chunk = std::move(input_queue.front());
                input_queue.pop();
            }

            // Process audio
            TranscriptionResult result;
            result.chunk_id = chunk.id;
            result.is_partial = false;
            try
            {
                result.segments = model.transcribe_raw_audio(chunk.data.data(), chunk.data.size());
            }
            catch (const std::exception &e)
            {
                std::cerr << "Exception during transcription: " << e.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "Unknown exception during transcription" << std::endl;
            }

            // Add result to output queue
            {
                std::lock_guard<std::mutex> lock(result_mutex);
                result_queue.push(result);
                result_cv.notify_one();
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
                        full_text += segment.text;
                    }
                    full_text = trim(full_text);
                    if (full_text.empty())
                        continue;

                    if (result_callback)
                    {
                        try
                        {
                            result_callback((int)result.chunk_id, result.segments, result.is_partial);
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

    std::string model_path;
    bool use_gpu;

    std::atomic<bool> running;
    std::atomic<size_t> next_chunk_id;
    size_t current_chunk_id;

    std::thread process_thread;
    std::thread result_thread;

    std::queue<AudioChunk> input_queue;
    std::mutex input_mutex;
    std::condition_variable input_cv;

    std::queue<TranscriptionResult> result_queue;
    std::mutex result_mutex;
    std::condition_variable result_cv;

    py::function result_callback;
};

class ThreadedWhisperModel : public AsyncWhisperModel
{
public:
    ThreadedWhisperModel(const std::string &model_path, bool use_gpu = false,
                         float max_duration_sec = 10.0f, int sample_rate = 16000)
        : AsyncWhisperModel(model_path, use_gpu),
          max_samples(static_cast<size_t>(max_duration_sec * sample_rate))
    {
    }

    ~ThreadedWhisperModel()
    {
        stop();
    }

    void setMaxDuration(float max_duration_sec, int sample_rate = 16000)
    {
        max_samples = static_cast<size_t>(max_duration_sec * sample_rate);
    }

    void start(py::function callback, int result_check_interval_ms = 100)
    {
        AsyncWhisperModel::start(callback, result_check_interval_ms);
    }

    void stop() override
    {
        AsyncWhisperModel::stop();

        // Clear accumulated buffer
        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            accumulated_buffer.clear();
        }
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
        std::vector<WhisperSegment> segments;
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
        for (const auto &segment : segments)
        {
            result.segments.push_back(segment);
        }
        // Set partial flag based on whether this is a final result
        result.is_partial = !(force_final || process_buffer.size() >= max_samples);

        // Add result to output queue
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            result_queue.push(result);
            result_cv.notify_one();
        }
    }

    void processThread() override
    {
        WhisperModel model(model_path, use_gpu);

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

    // Audio accumulation
    std::vector<float> accumulated_buffer;
    size_t max_samples;
    std::mutex buffer_mutex;
};

PYBIND11_MODULE(_whisper_cpp, m)
{
    // Bind WhisperToken
    py::class_<WhisperToken>(m, "WhisperToken")
        .def(py::init<>())
        .def_readwrite("id", &WhisperToken::id)
        .def_readwrite("p", &WhisperToken::p)
        .def_readwrite("t0", &WhisperToken::t0)
        .def_readwrite("t1", &WhisperToken::t1)
        .def_readwrite("text", &WhisperToken::text)
        .def("__str__", [](const WhisperToken &t)
             {
            std::stringstream ss;
            ss << t.text << " (id: " << t.id << ", p: " << t.p << ")";
            return ss.str(); });

    // Bind WhisperSement
    py::class_<WhisperSegment>(m, "WhisperSement")
        .def(py::init<>())
        .def_readwrite("text", &WhisperSegment::text)
        .def_readwrite("start", &WhisperSegment::start)
        .def_readwrite("end", &WhisperSegment::end)
        .def_readwrite("tokens", &WhisperSegment::tokens)
        .def("__str__", [](const WhisperSegment &s)
             { return s.text; })
        .def("__repr__", [](const WhisperSegment &s)
             {
            std::stringstream ss;
            ss << "WhisperSegment(text=\"" << s.text << "\", start=" << s.start << ", end=" << s.end << ")";
            return ss.str(); });

    // Expose synchronous model
    py::class_<WhisperModel>(m, "WhisperModel")
        .def(py::init<const std::string &, bool>())
        .def("transcribe", &WhisperModel::transcribe);

    // Expose asynchronous model
    py::class_<AsyncWhisperModel>(m, "AsyncWhisperModel")
        .def(py::init<const std::string &, bool>())
        .def("start", &AsyncWhisperModel::start,
             py::arg("callback"),
             py::arg("result_check_interval_ms") = 100)
        .def("stop", &AsyncWhisperModel::stop)
        .def("transcribe", &AsyncWhisperModel::transcribe)
        .def("queue_audio", &AsyncWhisperModel::queueAudio);

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
