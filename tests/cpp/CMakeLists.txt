include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(core_tests
    test_accounting.cpp
    test_execution.cpp
    test_env.cpp
    test_loader.cpp
)
target_link_libraries(core_tests PRIVATE trading_core GTest::gtest_main)
target_compile_definitions(core_tests PRIVATE TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}")

include(GoogleTest)
gtest_discover_tests(core_tests)
