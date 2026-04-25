#ifndef MARKET_DATA_LOADER_H
#define MARKET_DATA_LOADER_H

#include <string>
#include "MarketData.h"

/// @file MarketDataLoader.h
/// @brief Separates IO/parsing from environment logic.

constexpr char EXCEPTION_EXPECTED_TIMESTAMP[] = "CSV '{}': first column must be 'timestamp' but got '{}'";
constexpr char EXCEPTION_ROW_TOO_MANY_COL[] = "CSV '{}' row {}: too many columns";
constexpr char EXCEPTION_CANNOT_PARSE[] = "CSV '{}' row {}: cannot parse '{}' in column {}";
constexpr char EXCEPTION_NAN_INF[] = "CSV '{}' row {}: NaN or infinity in column {}";
constexpr char EXCEPTION_ROW_EXPECTED_N_COL[] = "CSV '{}' row {}: expected {} columns but found {}";
constexpr char EXCEPTION_CANNOT_OPEN_FILE[] = "Cannot open file: {}";
constexpr char EXCEPTION_FILE_EMPTY[] = "CSV '{}': file is empty";

/// @brief Utility class for loading market data from CSV files.
class MarketDataLoader {
public:
    MarketDataLoader() = delete;

    /// @brief Loads a CSV file into a MarketData object.
    /// @param path Path to the CSV file.
    /// @return Populated MarketData object.
    static MarketData load_csv(const std::string& path);

    /// @brief Parses a single CSV row.
    /// @param line CSV row string.
    /// @param row_number Current row number.
    /// @param path CSV file path.
    /// @param total_cols Expected number of columns.
    /// @return Parsed Bar object.
    static Bar parse_row(const std::string& line, int row_number, const std::string& path, int total_cols);
};

#endif 
