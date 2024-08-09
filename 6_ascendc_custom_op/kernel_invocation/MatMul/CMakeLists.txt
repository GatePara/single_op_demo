cmake_minimum_required(VERSION 3.16)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_STANDARD 17)

set(CCE_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CCE_CMAKE_PATH})
project(kernel_samples LANGUAGES CCE CXX)

add_subdirectory(cmake/cpu)
add_subdirectory(cmake/npu)

if(ASCEND_CUSTOM_TILING STREQUAL "CUSTOM_TILING")
    add_subdirectory(cmake/tiling)
endif()