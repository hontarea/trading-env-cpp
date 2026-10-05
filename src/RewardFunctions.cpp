#include <algorithm>
#include <cmath>

#include "RewardFunctions.h"

namespace RewardFunctions {

RewardFn log_return_reward() {
    return [](const Portfolio& previous,
              const Portfolio& current,
              const Bar& /*current_bar*/) -> double {
        const double ratio = current.equity() / previous.equity();
        return std::log(std::max(ratio, MIN_EQUITY_RATIO));
    };
}

}  // namespace RewardFunctions
