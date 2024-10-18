cmake_minimum_required(VERSION 3.15)
project(whisper_cpp_wrapper)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
if(DEFINED PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.6  # Specify a version/tag here
)
FetchContent_MakeAvailable(pybind11)

include(cmake/BuildWhispercpp.cmake)

# Include directories
include_directories(${Python_INCLUDE_DIRS})
include_directories(${Python_NumPy_INCLUDE_DIRS})

# Create the extension module
pybind11_add_module(_whisper_cpp src/whisper_wrapper.cpp)
target_link_libraries(_whisper_cpp PRIVATE Whispercpp)

# Set the output directory for the built module
set_target_properties(_whisper_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/simpler_whisper
)

# Copy the DLL to the output directory on Windows
if(WIN32)
    foreach(WHISPER_DLL ${WHISPER_DLLS})
        add_custom_command(TARGET _whisper_cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${WHISPER_DLL}"
            $<TARGET_FILE_DIR:_whisper_cpp>
        )
    endforeach()
endif()