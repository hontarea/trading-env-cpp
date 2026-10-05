// Runs a simple momentum rule through the environment and prints the first steps.
// Usage: smoke_test <csv_path>

#include <format>
#include <iostream>
#include <string>

#include "ExecutionModel.h"
#include "MarketDataLoader.h"
#include "RewardFunctions.h"
#include "TradingEnv.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: smoke_test <csv_path>\n";
        return 1;
    }

    try {
        auto data = std::make_shared<const MarketData>(MarketDataLoader::load_csv(argv[1]));
        std::cout << std::format("Loaded {} bars\n", data->size());

        // 1 bp half-spread, no impact, 5 bp commission.
        auto execution = std::make_shared<const ExecutionModel>(0.0001, 0.0, 0.0005);
        EnvConfig config;
        TradingEnv env(data, execution, RewardFunctions::log_return_reward(), config);
        auto obs = env.reset();

        std::cout << std::format("{:<6} {:>12} {:>6} {:>10} {:>12} {:>10} {:>12}\n",
            "Step", "LogReturn", "Act", "Exposure", "Reward", "Fee", "Equity");
        std::cout << std::string(74, '-') << "\n";

        int steps = 0;
        int trades = 0;
        double total_reward = 0.0;
        bool done = false;

        while (!done) {
            // Momentum rule: long after an up bar, flat otherwise.
            const double action = obs[0] > 0.0 ? 1.0 : 0.0;
            auto result = env.step(action);
            done = result.terminated || result.truncated;
            total_reward += result.reward;
            trades += result.info.units_traded != 0.0;

            if (steps < 10) {
                std::cout << std::format("{:<6} {:>12.6f} {:>6.1f} {:>10.3f} {:>12.6f} {:>10.4f} {:>12.2f}\n",
                    env.current_step(), obs[0], action, result.info.exposure, result.reward,
                    result.info.commission, result.info.equity);
            }
            ++steps;
            obs = std::move(result.observation);
        }

        const double final_equity = env.portfolio().equity();
        std::cout << "\n";
        std::cout << std::format("Steps:            {:>12}\n", steps);
        std::cout << std::format("Trades:           {:>12}\n", trades);
        std::cout << std::format("Final equity:     {:>12.2f}\n", final_equity);
        std::cout << std::format("Total return:     {:>12.4f}\n", final_equity / config.initial_cash - 1.0);
        std::cout << std::format("Sum of rewards:   {:>12.4f}  (= log(final/initial))\n", total_reward);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
