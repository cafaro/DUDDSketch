cmake_minimum_required(VERSION 3.16)
project(DISTRIBUTED_UDDSKETCH LANGUAGES C CXX)

# require c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

# IGRAPH dependencies
# find_package(igraph REQUIRED)

add_executable(dudds dudds.cpp uddsketch/uddsketch.cpp)

target_include_directories( dudds PUBLIC ${CMAKE_SOURCE_DIR}/uddsketch)
target_include_directories( dudds PUBLIC /usr/include/igraph)
target_link_libraries(dudds PUBLIC igraph)
