#include "Portfolio.h"

Portfolio::Portfolio(double initial_cash)
    : cash_(initial_cash), equity_(initial_cash) {}

void Portfolio::apply_fill(double units_delta, double fill_price, double commission) {
    cash_ -= units_delta * fill_price;
    cash_ -= commission;
    units_ += units_delta;
}

void Portfolio::mark_to_market(double price) {
    mark_price_ = price;
    equity_ = cash_ + units_ * price;
}

double Portfolio::exposure() const {
    return equity_ > 0.0 ? units_ * mark_price_ / equity_ : 0.0;
}
