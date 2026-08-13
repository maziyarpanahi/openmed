cmake_minimum_required(VERSION 3.14)
project(openmed_ffi LANGUAGES C)

option(
  OPENMED_FFI_ENABLE_TEST_SESSION
  "Build the synthetic token-classification session used by tests"
  OFF
)
set(
  OPENMED_ONNXRUNTIME_ROOT
  ""
  CACHE PATH
  "Extracted ONNX Runtime package containing headers and native libraries"
)

add_library(openmed_ffi SHARED openmed_ffi.c)
target_compile_features(openmed_ffi PRIVATE c_std_11)
target_compile_definitions(openmed_ffi PRIVATE OPENMED_FFI_BUILDING_LIBRARY)
target_include_directories(openmed_ffi PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

if(OPENMED_FFI_ENABLE_TEST_SESSION)
  target_compile_definitions(openmed_ffi PRIVATE OPENMED_FFI_ENABLE_TEST_SESSION)
endif()

if(OPENMED_ONNXRUNTIME_ROOT)
  target_compile_definitions(openmed_ffi PRIVATE OPENMED_FFI_ONNXRUNTIME)
  target_include_directories(
    openmed_ffi
    PRIVATE "${OPENMED_ONNXRUNTIME_ROOT}/headers"
  )
  if(ANDROID)
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(
      onnxruntime
      PROPERTIES IMPORTED_LOCATION
        "${OPENMED_ONNXRUNTIME_ROOT}/jni/${ANDROID_ABI}/libonnxruntime.so"
    )
    target_link_libraries(openmed_ffi PRIVATE onnxruntime)
  elseif(WIN32)
    target_link_libraries(
      openmed_ffi
      PRIVATE "${OPENMED_ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
    )
  else()
    find_library(
      OPENMED_ONNXRUNTIME_LIBRARY
      NAMES onnxruntime
      PATHS "${OPENMED_ONNXRUNTIME_ROOT}/lib"
      NO_DEFAULT_PATH
      REQUIRED
    )
    target_link_libraries(openmed_ffi PRIVATE "${OPENMED_ONNXRUNTIME_LIBRARY}")
  endif()
endif()

if(UNIX AND NOT APPLE)
  target_link_libraries(openmed_ffi PRIVATE m)
endif()

set_target_properties(
  openmed_ffi
  PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    OUTPUT_NAME openmed_ffi
)
