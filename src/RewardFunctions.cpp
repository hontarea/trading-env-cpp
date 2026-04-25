#include <cmath>

#include "RewardFunctions.h"

namespace RewardFunctions {

RewardFn log_return_reward() {
    return [](const Portfolio& previous,
              const Portfolio& current,
              const Bar& current_bar) -> double {
        double prev_equity = previous.equity();
        double curr_equity = current.equity();
        if (prev_equity <= 0.0) {
            return 0.0;
        }
        return std::log(curr_equity / prev_equity);
    };
}

} 
