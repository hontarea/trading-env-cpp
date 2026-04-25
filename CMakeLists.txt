cmake_minimum_required(VERSION 3.20)
project(TradingEnv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CORE_SOURCES
    src/ExecutionModel.cpp
    src/MarketData.cpp
    src/MarketDataLoader.cpp
    src/Portfolio.cpp
    src/RewardFunctions.cpp
    src/TradingEnv.cpp
)

add_library(trading_core STATIC ${CORE_SOURCES})
target_include_directories(trading_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(trading_test test.cpp)
target_link_libraries(trading_test PRIVATE trading_core)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

pybind11_add_module(trading_env_py bindings.cpp)
target_link_libraries(trading_env_py PRIVATE trading_core)

message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "pybind11 version: ${pybind11_VERSION}")
