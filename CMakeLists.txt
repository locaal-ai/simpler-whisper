cmake_minimum_required(VERSION 3.15)
project(whisper_cpp_wrapper)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch x86_64 -arch arm64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -arch x86_64 -arch arm64")
endif()

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6 # Specify a version/tag here
)
FetchContent_MakeAvailable(pybind11)

include(cmake/BuildWhispercpp.cmake)

# Create the extension module
pybind11_add_module(_whisper_cpp src/whisper_wrapper.cpp)
target_link_libraries(_whisper_cpp PRIVATE Whispercpp)

# Set the output directory for the built module
set_target_properties(
    _whisper_cpp PROPERTIES LIBRARY_OUTPUT_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/simpler_whisper)

# Copy the DLL to the output directory on Windows
if(WIN32 OR APPLE)
    foreach(WHISPER_ADDITIONAL_FILE ${WHISPER_ADDITIONAL_FILES})
        add_custom_command(
            TARGET _whisper_cpp
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${WHISPER_ADDITIONAL_FILE}" $<TARGET_FILE_DIR:_whisper_cpp>)
    endforeach()
endif()

if(APPLE)
    # Additional macOS-specific settings for the module
    set_target_properties(_whisper_cpp PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()
