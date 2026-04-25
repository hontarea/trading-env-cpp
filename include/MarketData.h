#ifndef MARKET_DATA_H
#define MARKET_DATA_H

#include <vector>
#include <unordered_map>
#include <string>

#include "Bar.h"

/// @file MarketData.h
/// @brief Sequential access to bars for simulation.

constexpr char FEATURE_TIMESTAMP[] = "timestamp";
constexpr char FEATURE_OPEN[] = "open";
constexpr char FEATURE_CLOSE[] = "close";
constexpr char FEATURE_FORECAST[] = "forecast";

/// @brief Container for a sequence of market bars and their feature mapping.
class MarketData {
public:
    /// @brief Constructor for MarketData.
    /// @param bars Vector of Bar objects.
    /// @param feature_map Mapping from feature name to its index in Bar::features.
    MarketData(std::vector<Bar> bars, std::unordered_map<std::string, size_t> feature_map);

    MarketData(const MarketData&) = delete;
    MarketData& operator=(const MarketData&) = delete;

    MarketData(MarketData&&)  = default;
    MarketData& operator=(MarketData&&)  = default;
    
    ~MarketData() = default;

    /// @brief Returns the number of bars.
    /// @return Number of bars.
    std::size_t size() const noexcept;

    /// @brief Access a bar at a specific index.
    /// @param index Bar index.
    /// @return Reference to Bar.
    const Bar& operator[](std::size_t index) const;

    /// @brief Access a bar at a specific index with bounds checking.
    /// @param index Bar index.
    /// @return Reference to Bar.
    const Bar& at(std::size_t index) const;

    /// @brief Returns pointer to the first bar.
    const Bar* begin() const;

    /// @brief Returns pointer to the past-the-end bar.
    const Bar* end() const;

    /// @brief Gets the index of a feature by name.
    /// @param name Name of the feature.
    /// @return Index of the feature.
    size_t get_feature_index(const std::string& name) const;

    /// @brief Returns the feature map.
    const std::unordered_map<std::string, size_t>& feature_map() const { return feature_map_; }

private:
    std::vector<Bar> bars_; 
    std::unordered_map<std::string, size_t> feature_map_;
};

#endif 
