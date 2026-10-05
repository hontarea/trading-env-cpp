#ifndef MARKET_DATA_H
#define MARKET_DATA_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Bar.h"

/// @file MarketData.h
/// @brief Sequential access to bars for simulation.

constexpr char FEATURE_TIMESTAMP[] = "timestamp";
constexpr char FEATURE_OPEN[] = "open";
constexpr char FEATURE_CLOSE[] = "close";
constexpr char FEATURE_FORECAST[] = "forecast";

constexpr char EXCEPTION_FEATURE_NOT_FOUND[] = "Feature not found: '{}'";
constexpr char EXCEPTION_BAR_WIDTH[] = "Bar {} has {} features, expected {}";
constexpr char EXCEPTION_TS_NOT_INCREASING[] = "Timestamps must be strictly increasing: bar {} ({}) is not after bar {} ({})";
constexpr char EXCEPTION_FEATURE_INDEX_RANGE[] = "Feature '{}' maps to index {}, but bars have only {} features";
constexpr char EXCEPTION_COLUMN_LENGTH[] = "Column '{}' has {} values, expected {} (one per timestamp)";
constexpr char EXCEPTION_COLUMN_DUPLICATE[] = "Duplicate column name '{}'";
constexpr char EXCEPTION_COLUMN_NOT_FINITE[] = "Column '{}' has a NaN or infinite value at row {}";
constexpr char EXCEPTION_NO_COLUMNS[] = "MarketData needs at least one feature column";

/// @brief Container for a sequence of market bars and their feature mapping.
/// Guarantees: every bar has the same width as the feature map, and timestamps are
/// strictly increasing.
class MarketData {
public:
    /// @brief Constructor for MarketData. Validates the invariants above.
    /// @param bars Vector of Bar objects.
    /// @param feature_map Mapping from feature name to its index in Bar::features.
    /// @throws DataLoadError if the invariants are violated.
    MarketData(std::vector<Bar> bars, std::unordered_map<std::string, std::size_t> feature_map);

    /// @brief Builds MarketData from in-memory columns (no CSV round-trip).
    /// @param timestamps One timestamp per bar.
    /// @param columns Ordered (name, values) pairs; each must have one value per timestamp.
    /// @throws DataLoadError on mismatched lengths, duplicate names or non-finite values.
    static MarketData from_columns(
        std::vector<std::int64_t> timestamps,
        const std::vector<std::pair<std::string, std::vector<double>>>& columns);

    MarketData(const MarketData&) = delete;
    MarketData& operator=(const MarketData&) = delete;

    MarketData(MarketData&&) = default;
    MarketData& operator=(MarketData&&) = default;

    ~MarketData() = default;

    /// @brief Returns the number of bars.
    std::size_t size() const noexcept;

    /// @brief Access a bar at a specific index (unchecked).
    const Bar& operator[](std::size_t index) const;

    /// @brief Access a bar at a specific index with bounds checking.
    const Bar& at(std::size_t index) const;

    /// @brief Returns pointer to the first bar.
    const Bar* begin() const;

    /// @brief Returns pointer to the past-the-end bar.
    const Bar* end() const;

    /// @brief Whether a feature with this name exists.
    bool has_feature(const std::string& name) const;

    /// @brief Gets the index of a feature by name.
    /// @throws ConfigError if the feature does not exist.
    std::size_t get_feature_index(const std::string& name) const;

    /// @brief Returns the feature map.
    const std::unordered_map<std::string, std::size_t>& feature_map() const { return feature_map_; }

private:
    std::vector<Bar> bars_;
    std::unordered_map<std::string, std::size_t> feature_map_;
};

#endif
