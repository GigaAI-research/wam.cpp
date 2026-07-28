cmake_minimum_required(VERSION 3.22)
project(wam_consumer LANGUAGES CXX)

find_package(wam 0.6 CONFIG REQUIRED)

add_executable(wam_consumer main.cpp)
target_link_libraries(wam_consumer PRIVATE wam::core)
target_compile_features(wam_consumer PRIVATE cxx_std_17)

enable_testing()
add_test(NAME wam_consumer_runs COMMAND wam_consumer)
