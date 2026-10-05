cmake_minimum_required(VERSION 3.20)
project(TradingEnv VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TRADING_ENV_BUILD_PYTHON "Build the pybind11 module" ON)
option(TRADING_ENV_BUILD_EXAMPLES "Build the C++ example programs" ON)
include(CTest)  # provides BUILD_TESTING (ON by default)

# Core library
add_library(trading_core STATIC
    src/ExecutionModel.cpp
    src/MarketData.cpp
    src/MarketDataLoader.cpp
    src/Portfolio.cpp
    src/RewardFunctions.cpp
    src/TradingEnv.cpp
)
target_include_directories(trading_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# The static library is linked into the Python extension (a shared object), so it must be PIC.
set_target_properties(trading_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(trading_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(TRADING_ENV_BUILD_EXAMPLES)
    add_executable(smoke_test examples/smoke_test.cpp)
    target_link_libraries(smoke_test PRIVATE trading_core)
endif()

# Python extension: installed as trading_env._core
if(TRADING_ENV_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_core bindings/bindings.cpp)
    target_link_libraries(_core PRIVATE trading_core)
    install(TARGETS _core LIBRARY DESTINATION trading_env)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests/cpp)
endif()
