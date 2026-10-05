#include <cmath>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "Exceptions.h"
#include "MarketData.h"
#include "MarketDataLoader.h"

namespace {

std::string trim(std::string_view s) {
    constexpr std::string_view whitespace = " \t\r\n";
    const auto first = s.find_first_not_of(whitespace);
    if (first == std::string_view::npos) {
        return {};
    }
    const auto last = s.find_last_not_of(whitespace);
    return std::string(s.substr(first, last - first + 1));
}

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream stream(line);
    std::string token;
    while (std::getline(stream, token, ',')) {
        tokens.push_back(trim(token));
    }
    // getline drops a trailing empty field ("a,b,"); keep it so the column count is honest.
    if (!line.empty() && line.back() == ',') {
        tokens.emplace_back();
    }
    return tokens;
}

}  // namespace

Bar MarketDataLoader::parse_row(const std::string& line, int row_number, const std::string& path, int total_cols) {
    const std::vector<std::string> tokens = split_csv(line);
    const int n_tokens = static_cast<int>(tokens.size());
    if (n_tokens > total_cols) {
        throw DataLoadError(std::format(EXCEPTION_ROW_TOO_MANY_COL, path, row_number));
    }
    if (n_tokens != total_cols) {
        throw DataLoadError(std::format(EXCEPTION_ROW_EXPECTED_N_COL, path, row_number, total_cols, n_tokens));
    }

    std::int64_t ts = 0;
    std::vector<double> features;
    features.reserve(total_cols - 1);

    for (int col = 0; col < total_cols; ++col) {
        const std::string& token = tokens[col];
        std::size_t pos = 0;
        try {
            if (col == 0) {
                ts = std::stoll(token, &pos);
            } else {
                const double val = std::stod(token, &pos);
                if (!std::isfinite(val)) {
                    throw DataLoadError(std::format(EXCEPTION_NAN_INF, path, row_number, col));
                }
                features.push_back(val);
            }
        } catch (const DataLoadError&) {
            throw;
        } catch (const std::exception&) {
            throw DataLoadError(std::format(EXCEPTION_CANNOT_PARSE, path, row_number, token, col));
        }
        // Reject trailing garbage such as "1.5abc", which stod/stoll would silently accept.
        if (pos != token.size()) {
            throw DataLoadError(std::format(EXCEPTION_CANNOT_PARSE, path, row_number, token, col));
        }
    }

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
    const std::vector<std::string> header = split_csv(line);
    if (header.empty() || header[0] != FEATURE_TIMESTAMP) {
        throw DataLoadError(std::format(EXCEPTION_EXPECTED_TIMESTAMP, path, header.empty() ? "" : header[0]));
    }
    if (header.size() < 2) {
        throw DataLoadError(std::format(EXCEPTION_NO_FEATURE_COLUMNS, path));
    }
    std::unordered_map<std::string, std::size_t> feature_map;
    for (std::size_t col = 1; col < header.size(); ++col) {
        if (header[col].empty()) {
            throw DataLoadError(std::format(EXCEPTION_EMPTY_COLUMN_NAME, path, col));
        }
        if (!feature_map.emplace(header[col], col - 1).second) {
            throw DataLoadError(std::format(EXCEPTION_DUPLICATE_COLUMN, path, header[col]));
        }
    }
    const int total_cols = static_cast<int>(header.size());

    std::vector<Bar> bars;
    int row_number = 1;

    while (std::getline(file, line)) {
        ++row_number;
        if (trim(line).empty()) {
            continue;
        }
        Bar bar = parse_row(line, row_number, path, total_cols);
        if (!bars.empty() && bar.timestamp <= bars.back().timestamp) {
            throw DataLoadError(std::format(EXCEPTION_ROW_TS_NOT_INCREASING,
                path, row_number, bar.timestamp, bars.back().timestamp));
        }
        bars.push_back(std::move(bar));
    }

    if (bars.empty()) {
        throw DataLoadError(std::format(EXCEPTION_NO_ROWS, path));
    }

    return MarketData(std::move(bars), std::move(feature_map));
}
