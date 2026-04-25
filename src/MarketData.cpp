#include "MarketData.h"
#include <stdexcept>

MarketData::MarketData(std::vector<Bar> bars, std::unordered_map<std::string, size_t> feature_map) 
    : bars_(std::move(bars)), feature_map_(std::move(feature_map)) {}

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

size_t MarketData::get_feature_index(const std::string& name) const {
    auto it = feature_map_.find(name);
    if (it == feature_map_.end()) {
        throw std::out_of_range("Feature not found: " + name);
    }
    return it->second;
}
