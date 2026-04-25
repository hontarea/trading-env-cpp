#ifndef PORTFOLIO_H
#define PORTFOLIO_H

#include <cstddef>

/// @file Portfolio.h
/// @brief Accurate accounting of cash/equity/position state.

/// @brief Manages cash, position, and equity state.
class Portfolio {
public:

    /// @brief Constructor for Portfolio.
    /// @param initial_cash Starting cash balance.
    Portfolio(double initial_cash) ;

    Portfolio(const Portfolio&) = default;
    Portfolio& operator=(const Portfolio&) = default;
    Portfolio(Portfolio&&) noexcept = default;
    Portfolio& operator=(Portfolio&&) noexcept = default;
    ~Portfolio() = default;

    /// @brief Updates state after an order fill.
    /// @param new_position New position size.
    /// @param fill_price Execution price.
    /// @param transaction_cost_log Log transaction cost.
    void apply_fill(int new_position, double fill_price, double transaction_cost_log);

    /// @brief Updates current equity.
    /// @param current_price Current market price.
    void update_equity(double current_price);

    /// @brief Advances the step counter.
    void step_forward();

    /// @brief Returns the current position.
    /// @return Position.
    int position() const;

    /// @brief Returns the current cash balance.
    /// @return Cash balance.
    double cash() const;

    /// @brief Returns the current total equity.
    /// @return Total equity.
    double equity() const;

    /// @brief Returns the entry price of the current position.
    /// @return Entry price.
    double entry_price() const;

    /// @brief Returns the current step index.
    /// @return Step index.
    std::size_t step_index() const;


private:
    int position_ = 0;         
    double cash_;               
    double equity_;            
    double entry_price_ = 0.0;  
    std::size_t step_index_ = 0; 
};

#endif 
