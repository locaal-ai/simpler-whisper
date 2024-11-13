include(ExternalProject)
include(FetchContent)

if(UNIX AND NOT APPLE)
  find_package(OpenMP REQUIRED)
  # Set compiler flags for OpenMP
  set(WHISPER_EXTRA_CXX_FLAGS "${WHISPER_EXTRA_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(WHISPER_EXTRA_C_FLAGS "${WHISPER_EXTRA_CXX_FLAGS} ${OpenMP_C_FLAGS}")
endif()

set(PREBUILT_WHISPERCPP_VERSION "0.0.7")
set(PREBUILT_WHISPERCPP_URL_BASE
    "https://github.com/locaal-ai/occ-ai-dep-whispercpp/releases/download/${PREBUILT_WHISPERCPP_VERSION}"
)

if(APPLE)
  # Store source directories for each architecture
  foreach(MACOS_ARCH IN ITEMS "x86_64" "arm64")
    if(${MACOS_ARCH} STREQUAL "x86_64")
      set(WHISPER_CPP_HASH
          "dc7fd5ff9c7fbb8623f8e14d9ff2872186cab4cd7a52066fcb2fab790d6092fc")
    elseif(${MACOS_ARCH} STREQUAL "arm64")
      set(WHISPER_CPP_HASH
          "ebed595ee431b182261bce41583993b149eed539e15ebf770d98a6bc85d53a92")
    endif()

    set(WHISPER_CPP_URL
        "${PREBUILT_WHISPERCPP_URL_BASE}/whispercpp-macos-${MACOS_ARCH}-${PREBUILT_WHISPERCPP_VERSION}.tar.gz"
    )

    # Use unique names for each architecture's fetch
    FetchContent_Declare(
      whispercpp_fetch_${MACOS_ARCH}
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      URL ${WHISPER_CPP_URL}
      URL_HASH SHA256=${WHISPER_CPP_HASH})
    FetchContent_MakeAvailable(whispercpp_fetch_${MACOS_ARCH})

    # Store the source dir for each arch
    if(${MACOS_ARCH} STREQUAL "x86_64")
      set(WHISPER_X86_64_DIR ${whispercpp_fetch_x86_64_SOURCE_DIR})
    else()
      set(WHISPER_ARM64_DIR ${whispercpp_fetch_arm64_SOURCE_DIR})
    endif()
  endforeach()

  # Create a directory for the universal binaries
  set(UNIVERSAL_LIB_DIR ${CMAKE_BINARY_DIR}/universal/lib)
  file(MAKE_DIRECTORY ${UNIVERSAL_LIB_DIR})

  # Create universal binaries using lipo
  execute_process(
    COMMAND
      lipo -create "${WHISPER_X86_64_DIR}/lib/libwhisper.a"
      "${WHISPER_ARM64_DIR}/lib/libwhisper.a" -output
      "${UNIVERSAL_LIB_DIR}/libwhisper.a")

  execute_process(
    COMMAND
      lipo -create "${WHISPER_X86_64_DIR}/lib/libggml.a"
      "${WHISPER_ARM64_DIR}/lib/libggml.a" -output
      "${UNIVERSAL_LIB_DIR}/libggml.a")

  execute_process(
    COMMAND
      lipo -create "${WHISPER_X86_64_DIR}/lib/libwhisper.coreml.a"
      "${WHISPER_ARM64_DIR}/lib/libwhisper.coreml.a" -output
      "${UNIVERSAL_LIB_DIR}/libwhisper.coreml.a")

  # Set up the imported libraries to use the universal binaries
  add_library(Whispercpp::Whisper STATIC IMPORTED)
  set_target_properties(
    Whispercpp::Whisper PROPERTIES IMPORTED_LOCATION
                                   "${UNIVERSAL_LIB_DIR}/libwhisper.a")
  set_target_properties(
    Whispercpp::Whisper PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                   ${WHISPER_ARM64_DIR}/include) # Either arch's
                                                                 # include dir
                                                                 # is fine

  add_library(Whispercpp::GGML STATIC IMPORTED)
  set_target_properties(
    Whispercpp::GGML PROPERTIES IMPORTED_LOCATION
                                "${UNIVERSAL_LIB_DIR}/libggml.a")

  add_library(Whispercpp::CoreML STATIC IMPORTED)
  set_target_properties(
    Whispercpp::CoreML PROPERTIES IMPORTED_LOCATION
                                  "${UNIVERSAL_LIB_DIR}/libwhisper.coreml.a")

  # Copy the metal file from either architecture (they should be identical)
  set(WHISPER_ADDITIONAL_FILES ${WHISPER_ARM64_DIR}/bin/ggml-metal.metal)
elseif(WIN32)
  if(NOT DEFINED ACCELERATION)
    message(
      FATAL_ERROR
        "ACCELERATION is not set. Please set it to either `cpu`, `cuda`, `vulkan` or `hipblas`"
    )
  endif()

  set(ARCH_PREFIX ${ACCELERATION})
  set(WHISPER_CPP_URL
      "${PREBUILT_WHISPERCPP_URL_BASE}/whispercpp-windows-${ARCH_PREFIX}-${PREBUILT_WHISPERCPP_VERSION}.zip"
  )
  if(${ACCELERATION} STREQUAL "cpu")
    set(WHISPER_CPP_HASH
        "c23862b4aac7d8448cf7de4d339a86498f88ecba6fa7d243bbd7fabdb13d4dd4")
    add_compile_definitions("LOCALVOCAL_WITH_CPU")
  elseif(${ACCELERATION} STREQUAL "cuda")
    set(WHISPER_CPP_HASH
        "a0adeaccae76fab0678d016a62b79a19661ed34eb810d8bae3b610345ee9a405")
    add_compile_definitions("LOCALVOCAL_WITH_CUDA")
  elseif(${ACCELERATION} STREQUAL "hipblas")
    set(WHISPER_CPP_HASH
        "bbad0b4eec01c5a801d384c03745ef5e97061958f8cf8f7724281d433d7d92a1")
    add_compile_definitions("LOCALVOCAL_WITH_HIPBLAS")
  elseif(${ACCELERATION} STREQUAL "vulkan")
    set(WHISPER_CPP_HASH
        "12bb34821f9efcd31f04a487569abff2b669221f2706fe0d09c17883635ef58a")
    add_compile_definitions("LOCALVOCAL_WITH_VULKAN")
  else()
    message(
      FATAL_ERROR
        "The ACCELERATION environment variable is not set to a valid value. Please set it to either `cpu` or `cuda` or `vulkan` or `hipblas`"
    )
  endif()

  FetchContent_Declare(
    whispercpp_fetch
    URL ${WHISPER_CPP_URL}
    URL_HASH SHA256=${WHISPER_CPP_HASH}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(whispercpp_fetch)

  add_library(Whispercpp::Whisper SHARED IMPORTED)
  set_target_properties(
    Whispercpp::Whisper
    PROPERTIES
      IMPORTED_LOCATION
      ${whispercpp_fetch_SOURCE_DIR}/bin/${CMAKE_SHARED_LIBRARY_PREFIX}whisper${CMAKE_SHARED_LIBRARY_SUFFIX}
  )
  set_target_properties(
    Whispercpp::Whisper
    PROPERTIES
      IMPORTED_IMPLIB
      ${whispercpp_fetch_SOURCE_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}whisper${CMAKE_STATIC_LIBRARY_SUFFIX}
  )
  set_target_properties(
    Whispercpp::Whisper PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                   ${whispercpp_fetch_SOURCE_DIR}/include)
  add_library(Whispercpp::GGML SHARED IMPORTED)
  set_target_properties(
    Whispercpp::GGML
    PROPERTIES
      IMPORTED_LOCATION
      ${whispercpp_fetch_SOURCE_DIR}/bin/${CMAKE_SHARED_LIBRARY_PREFIX}ggml${CMAKE_SHARED_LIBRARY_SUFFIX}
  )
  set_target_properties(
    Whispercpp::GGML
    PROPERTIES
      IMPORTED_IMPLIB
      ${whispercpp_fetch_SOURCE_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}ggml${CMAKE_STATIC_LIBRARY_SUFFIX}
  )

  if(${ACCELERATION} STREQUAL "cpu")
    # add openblas to the link line
    add_library(Whispercpp::OpenBLAS STATIC IMPORTED)
    set_target_properties(
      Whispercpp::OpenBLAS
      PROPERTIES IMPORTED_LOCATION
                 ${whispercpp_fetch_SOURCE_DIR}/lib/libopenblas.dll.a)
  endif()

  # glob all dlls in the bin directory and install them
  file(GLOB WHISPER_ADDITIONAL_FILES ${whispercpp_fetch_SOURCE_DIR}/bin/*.dll)
else()
  if(${CMAKE_BUILD_TYPE} STREQUAL Release OR ${CMAKE_BUILD_TYPE} STREQUAL
                                             RelWithDebInfo)
    set(Whispercpp_BUILD_TYPE Release)
  else()
    set(Whispercpp_BUILD_TYPE Debug)
  endif()
  set(Whispercpp_Build_GIT_TAG "v1.7.1")
  set(WHISPER_EXTRA_CXX_FLAGS "-fPIC")

  find_package(OpenMP REQUIRED)
  # Set compiler flags for OpenMP
  set(WHISPER_EXTRA_CXX_FLAGS "${WHISPER_EXTRA_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(WHISPER_EXTRA_C_FLAGS "${WHISPER_EXTRA_CXX_FLAGS} ${OpenMP_C_FLAGS}")

  # On Linux build a static Whisper library
  ExternalProject_Add(
    Whispercpp_Build
    DOWNLOAD_EXTRACT_TIMESTAMP true
    GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp.git
    GIT_TAG ${Whispercpp_Build_GIT_TAG}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config
                  ${Whispercpp_BUILD_TYPE}
    BUILD_BYPRODUCTS
      <INSTALL_DIR>/lib64/${CMAKE_STATIC_LIBRARY_PREFIX}whisper${CMAKE_STATIC_LIBRARY_SUFFIX}
      <INSTALL_DIR>/lib64/${CMAKE_STATIC_LIBRARY_PREFIX}ggml${CMAKE_STATIC_LIBRARY_SUFFIX}
    CMAKE_GENERATOR ${CMAKE_GENERATOR}
    INSTALL_COMMAND
      ${CMAKE_COMMAND} --install <BINARY_DIR> --config ${Whispercpp_BUILD_TYPE}
      && ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/ggml/include/ggml.h
      <INSTALL_DIR>/include
    CONFIGURE_COMMAND
      ${CMAKE_COMMAND} -E env ${WHISPER_ADDITIONAL_ENV} ${CMAKE_COMMAND}
      <SOURCE_DIR> -B <BINARY_DIR> -G ${CMAKE_GENERATOR}
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      -DCMAKE_BUILD_TYPE=${Whispercpp_BUILD_TYPE}
      -DCMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}
      -DCMAKE_OSX_DEPLOYMENT_TARGET=10.13
      -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES_}
      -DCMAKE_CXX_FLAGS=${WHISPER_EXTRA_CXX_FLAGS}
      -DCMAKE_C_FLAGS=${WHISPER_EXTRA_C_FLAGS} -DBUILD_SHARED_LIBS=OFF
      -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=OFF
      -DGGML_OPENMP=ON -WHISPER_BUILD_SERVER=OFF
      -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_HIPBLAS=OFF)

  ExternalProject_Get_Property(Whispercpp_Build INSTALL_DIR)

  # add the static Whisper library to the link line
  add_library(Whispercpp::Whisper STATIC IMPORTED)
  set_target_properties(
    Whispercpp::Whisper
    PROPERTIES
      IMPORTED_LOCATION
      ${INSTALL_DIR}/lib64/${CMAKE_STATIC_LIBRARY_PREFIX}whisper${CMAKE_STATIC_LIBRARY_SUFFIX}
  )
  add_library(Whispercpp::GGML STATIC IMPORTED)
  set_target_properties(
    Whispercpp::GGML
    PROPERTIES
      IMPORTED_LOCATION
      ${INSTALL_DIR}/lib64/${CMAKE_STATIC_LIBRARY_PREFIX}ggml${CMAKE_STATIC_LIBRARY_SUFFIX}
  )
  set_target_properties(
    Whispercpp::Whisper PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                   ${INSTALL_DIR}/include)
  set_property(
    TARGET Whispercpp::Whisper
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_CXX)

endif()

add_library(Whispercpp INTERFACE)
add_dependencies(Whispercpp Whispercpp_Build)
target_link_libraries(Whispercpp INTERFACE Whispercpp::Whisper Whispercpp::GGML)
if(WIN32 AND "${ACCELERATION}" STREQUAL "cpu")
  target_link_libraries(Whispercpp INTERFACE Whispercpp::OpenBLAS)
endif()
if(APPLE)
  target_link_libraries(
    Whispercpp
    INTERFACE "-framework Accelerate -framework CoreML -framework Metal")
  target_link_libraries(Whispercpp INTERFACE Whispercpp::CoreML)
endif(APPLE)
if(UNIX AND NOT APPLE)
  target_link_libraries(Whispercpp INTERFACE OpenMP::OpenMP_CXX)
endif()
