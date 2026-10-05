#include <cmath>
#include <format>

#include "Exceptions.h"
#include "MarketData.h"

MarketData::MarketData(std::vector<Bar> bars, std::unordered_map<std::string, std::size_t> feature_map)
    : bars_(std::move(bars)), feature_map_(std::move(feature_map)) {
    const std::size_t width = feature_map_.size();
    for (const auto& [name, index] : feature_map_) {
        if (index >= width) {
            throw DataLoadError(std::format(EXCEPTION_FEATURE_INDEX_RANGE, name, index, width));
        }
    }
    for (std::size_t i = 0; i < bars_.size(); ++i) {
        if (bars_[i].features.size() != width) {
            throw DataLoadError(std::format(EXCEPTION_BAR_WIDTH, i, bars_[i].features.size(), width));
        }
        if (i > 0 && bars_[i].timestamp <= bars_[i - 1].timestamp) {
            throw DataLoadError(std::format(EXCEPTION_TS_NOT_INCREASING,
                i, bars_[i].timestamp, i - 1, bars_[i - 1].timestamp));
        }
    }
}

MarketData MarketData::from_columns(
    std::vector<std::int64_t> timestamps,
    const std::vector<std::pair<std::string, std::vector<double>>>& columns) {
    if (columns.empty()) {
        throw DataLoadError(EXCEPTION_NO_COLUMNS);
    }
    const std::size_t n = timestamps.size();
    std::unordered_map<std::string, std::size_t> feature_map;
    for (std::size_t c = 0; c < columns.size(); ++c) {
        const auto& [name, values] = columns[c];
        if (values.size() != n) {
            throw DataLoadError(std::format(EXCEPTION_COLUMN_LENGTH, name, values.size(), n));
        }
        if (!feature_map.emplace(name, c).second) {
            throw DataLoadError(std::format(EXCEPTION_COLUMN_DUPLICATE, name));
        }
    }

    std::vector<Bar> bars;
    bars.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> features(columns.size());
        for (std::size_t c = 0; c < columns.size(); ++c) {
            const double value = columns[c].second[i];
            if (!std::isfinite(value)) {
                throw DataLoadError(std::format(EXCEPTION_COLUMN_NOT_FINITE, columns[c].first, i));
            }
            features[c] = value;
        }
        bars.emplace_back(timestamps[i], std::move(features));
    }
    return MarketData(std::move(bars), std::move(feature_map));
}

std::size_t MarketData::size() const noexcept {
    return bars_.size();
}

const Bar& MarketData::operator[](std::size_t index) const {
    return bars_[index];
}

const Bar& MarketData::at(std::size_t index) const {
    return bars_.at(index);
}

const Bar* MarketData::begin() const {
    return bars_.data();
}

const Bar* MarketData::end() const {
    return bars_.data() + bars_.size();
}

bool MarketData::has_feature(const std::string& name) const {
    return feature_map_.contains(name);
}

std::size_t MarketData::get_feature_index(const std::string& name) const {
    auto it = feature_map_.find(name);
    if (it == feature_map_.end()) {
        throw ConfigError(std::format(EXCEPTION_FEATURE_NOT_FOUND, name));
    }
    return it->second;
}
