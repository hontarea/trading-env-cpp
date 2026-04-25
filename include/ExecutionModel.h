#ifndef EXECUTION_MODEL_H
#define EXECUTION_MODEL_H

/// @file ExecutionModel.h
/// @brief Simulates real-world market friction (slippage/transaction costs).

// Execution Model
constexpr int POSITION_FLAT = 0;
constexpr double MAX_POSITION_CHANGE = 2.0;
constexpr int OPEN_LONG = 1;
constexpr int OPEN_SHORT = -1;

constexpr char EXCEPTION_HALF_SPREAD_NEG[] = "half_spread must be non-negative, got {}";
constexpr char EXCEPTION_IMPACT_COEF_NEG[] = "impact_coef must be non-negative, got {}";
constexpr char EXCEPTION_TXN_COST_RATE_NEG[] = "txn_cost_rate must be non-negative, got {}";
constexpr char EXCEPTION_TXN_COST_RATE_TOO_HIGH[] = "txn_cost_rate * {} must be < 1.0 (max |delta_pos| is {}), got {}";

/// @brief Handles price slippage and transaction costs.
class ExecutionModel {
public:

    /// @brief Constructor for ExecutionModel.
    /// @param half_spread Fixed cost as half of the bid-ask spread.
    /// @param impact_coef Coefficient for market impact (slippage).
    /// @param txn_cost_rate Rate for transaction costs.
    ExecutionModel(double half_spread, double impact_coef, double txn_cost_rate);

    /// @brief Computes the price after slippage.
    /// @param mid_price Mid-market price.
    /// @param position_change Change in position.
    /// @return Calculated fill price.
    double compute_fill_price(double mid_price, int position_change) const;

    /// @brief Computes transaction costs in log terms.
    /// @param position_change Change in position size.
    /// @return Calculated transaction cost.
    double compute_transaction_cost_log(int position_change) const;

private:
    double half_spread_;    
    double impact_coef_;    
    double txn_cost_rate_;  
};

#endif 
 