#ifndef PORTFOLIO_H
#define PORTFOLIO_H

/// @file Portfolio.h
/// @brief Exact accounting of cash / position / equity.

/// @brief Manages cash, position and equity for a single asset.
///
/// Invariant: after every mark_to_market(p), equity() == cash() + units() * p.
/// Short positions are booked symmetrically: selling adds the proceeds to cash and makes
/// units negative, so a falling price increases equity.
class Portfolio {
public:
    /// @brief Constructor for Portfolio.
    /// @param initial_cash Starting cash balance.
    explicit Portfolio(double initial_cash);

    /// @brief Books a fill: cash -= units_delta * fill_price + commission.
    /// @param units_delta Signed change in position (units).
    /// @param fill_price Execution price.
    /// @param commission Commission in currency.
    void apply_fill(double units_delta, double fill_price, double commission);

    /// @brief Revalues the position at the given price and updates equity.
    /// @param price Current market price.
    void mark_to_market(double price);

    /// @brief Equity if the position were valued at `price` (does not change state).
    double equity_at(double price) const { return cash_ + units_ * price; }

    /// @brief Current position in units (negative = short).
    double units() const { return units_; }

    /// @brief Current cash balance (may be negative when leveraged long).
    double cash() const { return cash_; }

    /// @brief Equity at the last mark.
    double equity() const { return equity_; }

    /// @brief Price used for the last mark.
    double mark_price() const { return mark_price_; }

    /// @brief Signed position value as a fraction of equity at the last mark (0 if equity <= 0).
    double exposure() const;

private:
    double units_ = 0.0;
    double cash_;
    double equity_;
    double mark_price_ = 0.0;
};

#endif
