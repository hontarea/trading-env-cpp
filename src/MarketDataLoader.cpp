#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <format>
#include <unordered_map>

#include "Exceptions.h"
#include "MarketData.h"
#include "MarketDataLoader.h"

Bar MarketDataLoader::parse_row(const std::string& line, int row_number, const std::string& path, int total_cols) {
    std::istringstream stream(line);
    std::string token;
    std::vector<double> values;
    values.reserve(total_cols);
    int col = 0;

    while (std::getline(stream, token, ',')) {
        if (col >= total_cols) {
            throw DataLoadError(std::format(EXCEPTION_ROW_TOO_MANY_COL, path, row_number));
        }

        std::size_t pos = 0;
        double val = 0.0;
        try {
            val = std::stod(token, &pos);
        } catch (const std::exception&) {
            throw DataLoadError(std::format(EXCEPTION_CANNOT_PARSE, path, row_number, token, col));
        }

        if (std::isnan(val) || std::isinf(val)) {
            throw DataLoadError(std::format(EXCEPTION_NAN_INF, path, row_number, col));
        }

        values.push_back(val);
        ++col;
    }

    if (col != total_cols) {
        throw DataLoadError(std::format(EXCEPTION_ROW_EXPECTED_N_COL, path, row_number, total_cols, col));
    }

    std::int64_t ts = static_cast<std::int64_t>(values[0]);
    std::vector<double> features(values.begin() + 1, values.end());
    return Bar{ts, std::move(features)};
}

MarketData MarketDataLoader::load_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw DataLoadError(std::format(EXCEPTION_CANNOT_OPEN_FILE, path));
    }

    std::string line;

    if (!std::getline(file, line)) {
        throw DataLoadError(std::format(EXCEPTION_FILE_EMPTY, path));
    }
    
    // Parse header
    std::unordered_map<std::string, size_t> feature_map;
    std::istringstream h_stream(line);
    std::string token;
    int col = 0;
    while (std::getline(h_stream, token, ',')) {
        if (col == 0) {
            if (token != FEATURE_TIMESTAMP) {
                throw DataLoadError(std::format(EXCEPTION_EXPECTED_TIMESTAMP, path, token));
            }
        } else {
            feature_map[token] = col - 1;
        }
        col++;
    }
    int total_cols = col;

    std::vector<Bar> bars;
    int row_number = 1;

    while (std::getline(file, line)) {
        ++row_number;

        bars.push_back(parse_row(line, row_number, path, total_cols));
    }

    return MarketData(std::move(bars), std::move(feature_map));
}
