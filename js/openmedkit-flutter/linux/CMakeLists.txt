cmake_minimum_required(VERSION 3.13)
set(PROJECT_NAME "openmedkit_flutter")
project(${PROJECT_NAME} LANGUAGES C CXX)

set(PLUGIN_NAME "openmedkit_flutter_plugin")

add_library(${PLUGIN_NAME} SHARED
  "open_med_kit_flutter_plugin.cc"
)
apply_standard_settings(${PLUGIN_NAME})
set_target_properties(${PLUGIN_NAME} PROPERTIES
  CXX_VISIBILITY_PRESET hidden
)
target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)
target_include_directories(${PLUGIN_NAME} INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(${PLUGIN_NAME} PRIVATE flutter)
target_link_libraries(${PLUGIN_NAME} PRIVATE PkgConfig::GTK)

add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../native"
  "${CMAKE_CURRENT_BINARY_DIR}/openmed_ffi"
)
target_link_libraries(${PLUGIN_NAME} PRIVATE openmed_ffi)

set(openmedkit_flutter_bundled_libraries
  "$<TARGET_FILE:openmed_ffi>"
  PARENT_SCOPE
)
