cmake_minimum_required(VERSION 3.20)

project(optioryx_order_batching VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

find_package(Python3 COMPONENTS Interpreter)

add_library(
    optioryx_objects OBJECT
    src/algorithms/common.cpp
    src/core/evaluator.cpp
    src/core/analysis.cpp
    src/io/instance_loader.cpp
    src/io/submission_io.cpp
    src/io/config_loader.cpp
    src/util/output.cpp
    src/algorithms/greedy_solver.cpp
    src/algorithms/simulated_annealing_solver.cpp
    src/algorithms/genetic_solver.cpp
    src/algorithms/seed_selection_solver.cpp
)

target_include_directories(optioryx_objects PUBLIC include)

if(MSVC)
    target_compile_options(optioryx_objects PRIVATE /W4 /permissive-)
else()
    target_compile_options(optioryx_objects PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

add_executable(optioryx_cli src/main.cpp $<TARGET_OBJECTS:optioryx_objects>)
target_include_directories(optioryx_cli PRIVATE include)
target_compile_definitions(optioryx_cli PRIVATE OPTIORYX_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_executable(optioryx_tests tests/test_main.cpp $<TARGET_OBJECTS:optioryx_objects>)
target_include_directories(optioryx_tests PRIVATE include)
target_compile_definitions(optioryx_tests PRIVATE OPTIORYX_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

if(BUILD_TESTING)
    add_test(NAME unit_tests COMMAND optioryx_tests)
    set_tests_properties(unit_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    if(Python3_Interpreter_FOUND)
        add_test(
            NAME benchmark_smoke
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tests/smoke_test.py
                --cli $<TARGET_FILE:optioryx_cli>
                --instance ${CMAKE_SOURCE_DIR}/fixtures/small.instance
                --workdir ${CMAKE_BINARY_DIR}/smoke
        )
        set_tests_properties(benchmark_smoke PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()
endif()
