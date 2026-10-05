#ifndef BAR_H
#define BAR_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

/// @file Bar.h
/// @brief Standard unit of market price data.

/// @brief Represents a single market bar: a timestamp plus an arbitrary set of features
/// (OHLCV, forecasts, indicators, ...). Feature names are resolved by MarketData.
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

    double get_feature(std::size_t index) const { return features[index]; }
};

/// @brief Computes the log return of a feature (typically close) between two bars.
/// @param current Earlier bar.
/// @param next Later bar.
/// @param price_idx Index of the price feature.
/// @return Log return.
inline double log_return(const Bar& current, const Bar& next, std::size_t price_idx) {
    return std::log(next.get_feature(price_idx) / current.get_feature(price_idx));
}

#endif
