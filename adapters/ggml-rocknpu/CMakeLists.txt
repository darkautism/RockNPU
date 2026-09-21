cmake_minimum_required(VERSION 3.16)
project(ggml-rocknpu LANGUAGES CXX)

# This adapter sits on every decode projection. An unset single-config build
# type otherwise produces an unoptimized plugin even though Cargo uses Release.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(GGML_SOURCE_DIR "" CACHE PATH "Path to a llama.cpp ggml directory")
set(GGML_CPU_LIBRARY "" CACHE FILEPATH "Path to the matching llama.cpp libggml-cpu.so")
if(NOT GGML_SOURCE_DIR)
    message(FATAL_ERROR "Set GGML_SOURCE_DIR to <llama.cpp>/ggml")
endif()
if(NOT GGML_CPU_LIBRARY)
    message(FATAL_ERROR "Set GGML_CPU_LIBRARY to the libggml-cpu.so from the same llama.cpp build")
endif()

get_filename_component(ROCKNPU_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
find_program(CARGO_EXECUTABLE cargo REQUIRED)
set(ROCKNPU_CAPI_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/rust-target")
set(ROCKNPU_CAPI_LIBRARY "${ROCKNPU_CAPI_TARGET_DIR}/release/librocknpu_capi.so")

add_custom_target(rocknpu-capi-build
    COMMAND "${CARGO_EXECUTABLE}" build --release -p rocknpu-capi --target-dir "${ROCKNPU_CAPI_TARGET_DIR}"
    WORKING_DIRECTORY "${ROCKNPU_SOURCE_DIR}"
    COMMENT "Building RockNPU C ABI bridge"
    VERBATIM
)

add_library(ggml-rocknpu SHARED src/ggml-rocknpu.cpp)
add_dependencies(ggml-rocknpu rocknpu-capi-build)
target_compile_features(ggml-rocknpu PRIVATE cxx_std_17)
target_compile_definitions(ggml-rocknpu PRIVATE GGML_BACKEND_DL)
target_include_directories(ggml-rocknpu PRIVATE
    "${GGML_SOURCE_DIR}/include"
    "${GGML_SOURCE_DIR}/src"
    "${ROCKNPU_SOURCE_DIR}/crates/rocknpu-capi/include"
)
target_link_directories(ggml-rocknpu PRIVATE "${ROCKNPU_CAPI_TARGET_DIR}/release")
target_link_libraries(ggml-rocknpu PRIVATE rocknpu_capi "${GGML_CPU_LIBRARY}")
set_target_properties(ggml-rocknpu PROPERTIES
    OUTPUT_NAME "ggml-rocknpu"
    BUILD_RPATH "$ORIGIN"
)
add_custom_command(TARGET ggml-rocknpu POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${ROCKNPU_CAPI_LIBRARY}"
        "$<TARGET_FILE_DIR:ggml-rocknpu>/librocknpu_capi.so"
)
