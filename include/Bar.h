#ifndef BAR_H
#define BAR_H

#include <vector>
#include <cmath>
#include <cstdint>

/// @file Bar.h
/// @brief Standard unit of market price data.

/// @brief Represents a single market bar.
struct Bar {
    std::int64_t timestamp = 0; 
    std::vector<double> features;

    /// @brief Default constructor.
    Bar() = default;

    /// @brief Parameterized constructor.
    /// @param ts Timestamp.
    /// @param feats Vector of feature values.
    Bar(std::int64_t ts, std::vector<double> feats) 
        : timestamp(ts), features(std::move(feats)) {}

    double get_feature(size_t index) const { return features[index]; }
};

/// @brief Computes log return between two bars.
/// @param current Current bar.
/// @param next Next bar.
/// @param close_idx Index of the close price feature.
/// @return Log return.
inline double log_return(const Bar& current, const Bar& next, size_t close_idx) {
    return std::log(next.get_feature(close_idx) / current.get_feature(close_idx));
}
#endif 
