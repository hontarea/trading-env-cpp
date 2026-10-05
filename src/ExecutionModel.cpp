#include <cmath>
#include <format>

#include "Exceptions.h"
#include "ExecutionModel.h"

ExecutionModel::ExecutionModel(double half_spread, double impact_coef, double commission_rate)
    : half_spread_(half_spread), impact_coef_(impact_coef), commission_rate_(commission_rate) {
    if (!(half_spread_ >= 0.0 && half_spread_ < 1.0)) {
        throw ConfigError(std::format(EXCEPTION_HALF_SPREAD_RANGE, half_spread_));
    }
    if (!(impact_coef_ >= 0.0) || !std::isfinite(impact_coef_)) {
        throw ConfigError(std::format(EXCEPTION_IMPACT_COEF_NEG, impact_coef_));
    }
    if (!(commission_rate_ >= 0.0 && commission_rate_ < 1.0)) {
        throw ConfigError(std::format(EXCEPTION_COMMISSION_RATE_RANGE, commission_rate_));
    }
}

double ExecutionModel::fill_price(double mid_price, double units_delta, double delta_exposure) const {
    if (units_delta == 0.0) {
        return mid_price;
    }
    const double sign = (units_delta > 0.0) ? 1.0 : -1.0;
    return mid_price * (1.0 + sign * (half_spread_ + impact_coef_ * std::abs(delta_exposure)));
}

double ExecutionModel::commission(double units_delta, double fill_price) const {
    return commission_rate_ * std::abs(units_delta) * fill_price;
}
