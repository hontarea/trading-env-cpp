#include "Portfolio.h"

Portfolio::Portfolio(double initial_cash)
    : cash_(initial_cash), equity_(initial_cash) {}

void Portfolio::apply_fill(int new_position, double fill_price, double transaction_cost_log) {
    int delta = new_position - position_;
    if (delta != 0) {
        cash_ -= static_cast<double>(delta) * fill_price;
        cash_ -= transaction_cost_log;
    }
    position_ = new_position;
    entry_price_ = (new_position != 0) ? fill_price : 0.0;
}

void Portfolio::update_equity(double current_price) {
    equity_ = cash_ + static_cast<double>(position_) * current_price;
}

void Portfolio::step_forward() {
    ++step_index_;
}

int Portfolio::position() const { return position_; }
double Portfolio::cash() const { return cash_; }
double Portfolio::equity() const { return equity_; }
double Portfolio::entry_price() const { return entry_price_; }
std::size_t Portfolio::step_index() const { return step_index_; }
