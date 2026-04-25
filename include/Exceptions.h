#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H


#include <stdexcept>

/// @file Exceptions.h
/// @brief Distinctive error handling.

/// @brief Base class for market simulation errors.
/// Provides a common base for all custom exceptions.
class MarketSimError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// @brief Error during data loading.
/// Differentiates IO/parsing errors from other types.
class DataLoadError : public MarketSimError {
    using MarketSimError::MarketSimError;
};

/// @brief Error in configuration.
/// Differentiates configuration errors from other types.
class ConfigError : public MarketSimError {
    using MarketSimError::MarketSimError;
};

#endif 
 