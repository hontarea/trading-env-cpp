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
        auto data = std::make_shared<const MarketData>(
            MarketDataLoader::load_csv(argv[1]));

        std::cout << std::format("Loaded {} bars\n", data->size());

        auto execution = std::make_shared<ExecutionModel>(0.0005, 0.0001, 0.001);
        auto reward_fn = RewardFunctions::log_return_reward();
        EnvConfig config;

        TradingEnv env(data, std::move(execution), reward_fn, config);
        auto obs = env.reset();

        std::cout << std::format("{:<6} {:>12} {:>12} {:>4} {:>4} {:>12} {:>12} {:>12} {:>10}\n",
            "Step", "LogReturn", "SMA", "Pos", "Act", "Reward", "Equity", "Forecast", "Remain");
        std::cout << std::string(105, '-') << "\n";

        int trades = 0;
        int steps_shown = 0;
        double total_reward = 0.0;
        bool terminated = false;

        while (!terminated) {
            int action = (obs.last_log_return > 0.0) ? OPEN_LONG : POSITION_FLAT;
            auto result = env.step(action);
            terminated = result.terminated;

            if (action != obs.current_position) {
                ++trades;
            }

            total_reward += result.reward;

            if (steps_shown < 20) {
                std::cout << std::format("{:<6} {:>12.6f} {:>12.6f} {:>4} {:>4} {:>12.6f} {:>12.2f} {:>12.6f} {:>10}\n",
                    env.current_step(), obs.last_log_return, obs.short_window_mean_return,
                    obs.current_position, action, result.reward, env.portfolio().equity(),
                    obs.forecast, obs.steps_remaining);
                ++steps_shown;
            }

            obs = result.observation;
        }

        double final_equity = env.portfolio().equity();
        double total_return = (final_equity - config.initial_cash) / config.initial_cash;

        std::cout << "\n";
        std::cout << std::format("Final equity:   {:>12.2f}\n", final_equity);
        std::cout << std::format("Total return:   {:>12.4f}\n", total_return);
        std::cout << std::format("Position changes: {}\n", trades);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
