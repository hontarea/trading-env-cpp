#ifndef EXECUTION_MODEL_H
#define EXECUTION_MODEL_H

/// @file ExecutionModel.h
/// @brief Simulates real-world market friction (spread, market impact, commission).

constexpr char EXCEPTION_HALF_SPREAD_RANGE[] = "half_spread must be in [0, 1), got {}";
constexpr char EXCEPTION_IMPACT_COEF_NEG[] = "impact_coef must be non-negative, got {}";
constexpr char EXCEPTION_COMMISSION_RATE_RANGE[] = "commission_rate must be in [0, 1), got {}";

/// @brief Turns an order into a fill price and a commission.
///
/// For an order that changes the position by `units_delta` units at mid price `mid`:
///   fill_price = mid * (1 + sign(units_delta) * (half_spread + impact_coef * |delta_exposure|))
///   commission = commission_rate * |units_delta| * fill_price            (in currency)
/// where delta_exposure = units_delta * mid / equity is the traded notional as a fraction of
/// equity. Expressing impact in exposure (not units) keeps costs independent of the asset's
/// price level and the account size.
class ExecutionModel {
public:
    /// @brief Constructor for ExecutionModel.
    /// @param half_spread Half of the bid-ask spread, as a fraction of the mid price.
    /// @param impact_coef Linear market-impact coefficient per unit of traded exposure.
    /// @param commission_rate Commission as a fraction of traded notional.
    /// @throws ConfigError on out-of-range parameters.
    ExecutionModel(double half_spread, double impact_coef, double commission_rate);

    /// @brief Computes the execution price including spread and impact.
    /// @param mid_price Mid-market price at execution time.
    /// @param units_delta Signed change in position (units); positive = buy.
    /// @param delta_exposure |units_delta| * mid_price / equity.
    /// @return Fill price (equals mid_price when nothing is traded).
    double fill_price(double mid_price, double units_delta, double delta_exposure) const;

    /// @brief Computes the commission in currency.
    /// @param units_delta Signed change in position (units).
    /// @param fill_price Execution price.
    /// @return Commission, always >= 0.
    double commission(double units_delta, double fill_price) const;

    double half_spread() const { return half_spread_; }
    double impact_coef() const { return impact_coef_; }
    double commission_rate() const { return commission_rate_; }

private:
    double half_spread_;
    double impact_coef_;
    double commission_rate_;
};

#endif
