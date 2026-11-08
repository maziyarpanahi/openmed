cmake_minimum_required(VERSION 3.14)
project(openmed_ffi LANGUAGES C)

option(
  OPENMED_FFI_ENABLE_TEST_SESSION
  "Build the synthetic token-classification session used by tests"
  OFF
)
option(
  OPENMED_FFI_BUILD_TESTS
  "Build the native checks that run against the synthetic fixture session"
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
  # Android AARs ship headers/; desktop release archives ship include/.
  target_include_directories(
    openmed_ffi
    PRIVATE
      "${OPENMED_ONNXRUNTIME_ROOT}/headers"
      "${OPENMED_ONNXRUNTIME_ROOT}/include"
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
if(UNIX AND NOT APPLE AND NOT ANDROID)
  # The runtime ownership registry uses a pthread mutex.
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  target_link_libraries(openmed_ffi PRIVATE Threads::Threads)
endif()

set_target_properties(
  openmed_ffi
  PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    OUTPUT_NAME openmed_ffi
)

if(OPENMED_FFI_BUILD_TESTS)
  enable_testing()
  add_executable(openmed_ffi_tests tests/openmed_ffi_test.c)
  target_compile_features(openmed_ffi_tests PRIVATE c_std_11)
  target_compile_definitions(
    openmed_ffi_tests
    PRIVATE OPENMED_FFI_BUILDING_LIBRARY OPENMED_FFI_ENABLE_TEST_SESSION
  )
  if(UNIX AND NOT APPLE AND NOT ANDROID)
    target_link_libraries(openmed_ffi_tests PRIVATE m Threads::Threads)
  endif()
  add_test(NAME openmed_ffi_native_checks COMMAND openmed_ffi_tests)
endif()
