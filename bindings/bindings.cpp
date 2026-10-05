#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Bar.h"
#include "Exceptions.h"
#include "ExecutionModel.h"
#include "MarketData.h"
#include "MarketDataLoader.h"
#include "Portfolio.h"
#include "RewardFunctions.h"
#include "TradingEnv.h"

namespace py = pybind11;

namespace {

py::array_t<double> to_numpy(const std::vector<double>& values) {
    py::array_t<double> out(static_cast<py::ssize_t>(values.size()));
    std::copy(values.begin(), values.end(), out.mutable_data());
    return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ trading environment core (pybind11 bindings)";

    // Exceptions: keep the C++ hierarchy so Python code can catch by type.
    auto market_sim_error = py::register_exception<MarketSimError>(m, "MarketSimError", PyExc_RuntimeError);
    py::register_exception<DataLoadError>(m, "DataLoadError", market_sim_error);
    py::register_exception<ConfigError>(m, "ConfigError", market_sim_error);
    py::register_exception<InvalidActionError>(m, "InvalidActionError", market_sim_error);
    py::register_exception<EpisodeError>(m, "EpisodeError", market_sim_error);

    // Constants
    m.attr("DEFAULT_INITIAL_CASH") = DEFAULT_INITIAL_CASH;
    m.attr("DEFAULT_MAX_LEVERAGE") = DEFAULT_MAX_LEVERAGE;
    m.attr("SMA_WINDOW_SIZE") = SMA_WINDOW_SIZE;
    m.attr("FEATURE_TIMESTAMP") = FEATURE_TIMESTAMP;
    m.attr("FEATURE_OPEN") = FEATURE_OPEN;
    m.attr("FEATURE_CLOSE") = FEATURE_CLOSE;
    m.attr("FEATURE_FORECAST") = FEATURE_FORECAST;

    // Structs
    py::class_<Bar>(m, "Bar")
        .def(py::init<>())
        .def(py::init<std::int64_t, std::vector<double>>())
        .def_readwrite("timestamp", &Bar::timestamp)
        .def_readwrite("features", &Bar::features)
        .def("get_feature", &Bar::get_feature);

    py::class_<EnvConfig>(m, "EnvConfig")
        .def(py::init<>())
        .def_readwrite("initial_cash", &EnvConfig::initial_cash)
        .def_readwrite("start_index", &EnvConfig::start_index)
        .def_readwrite("end_index", &EnvConfig::end_index)
        .def_readwrite("max_leverage", &EnvConfig::max_leverage)
        .def_readwrite("obs_features", &EnvConfig::obs_features)
        .def_readwrite("obs_lookback", &EnvConfig::obs_lookback);

    py::class_<StepInfo>(m, "StepInfo")
        .def_readonly("fill_price", &StepInfo::fill_price)
        .def_readonly("units_traded", &StepInfo::units_traded)
        .def_readonly("commission", &StepInfo::commission)
        .def_readonly("equity", &StepInfo::equity)
        .def_readonly("exposure", &StepInfo::exposure)
        .def_readonly("bankrupt", &StepInfo::bankrupt)
        .def("to_dict", [](const StepInfo& i) {
            py::dict d;
            d["fill_price"] = i.fill_price;
            d["units_traded"] = i.units_traded;
            d["commission"] = i.commission;
            d["equity"] = i.equity;
            d["exposure"] = i.exposure;
            d["bankrupt"] = i.bankrupt;
            return d;
        });

    py::class_<StepResult>(m, "StepResult")
        .def_property_readonly("observation", [](const StepResult& r) { return to_numpy(r.observation); })
        .def_readonly("reward", &StepResult::reward)
        .def_readonly("terminated", &StepResult::terminated)
        .def_readonly("truncated", &StepResult::truncated)
        .def_readonly("info", &StepResult::info);

    // Core Classes
    py::class_<ExecutionModel, std::shared_ptr<ExecutionModel>>(m, "ExecutionModel")
        .def(py::init<double, double, double>(),
             py::arg("half_spread") = 0.0, py::arg("impact_coef") = 0.0, py::arg("commission_rate") = 0.0)
        .def("fill_price", &ExecutionModel::fill_price,
             py::arg("mid_price"), py::arg("units_delta"), py::arg("delta_exposure"))
        .def("commission", &ExecutionModel::commission, py::arg("units_delta"), py::arg("fill_price"))
        .def_property_readonly("half_spread", &ExecutionModel::half_spread)
        .def_property_readonly("impact_coef", &ExecutionModel::impact_coef)
        .def_property_readonly("commission_rate", &ExecutionModel::commission_rate);

    py::class_<Portfolio>(m, "Portfolio")
        .def_property_readonly("units", &Portfolio::units)
        .def_property_readonly("cash", &Portfolio::cash)
        .def_property_readonly("equity", &Portfolio::equity)
        .def_property_readonly("mark_price", &Portfolio::mark_price)
        .def_property_readonly("exposure", &Portfolio::exposure);

    py::class_<MarketData, std::shared_ptr<MarketData>>(m, "MarketData")
        .def_static("from_columns",
            [](std::vector<std::int64_t> timestamps, const py::dict& columns) {
                // Python dicts are ordered, so column order is preserved.
                std::vector<std::pair<std::string, std::vector<double>>> cols;
                for (auto item : columns) {
                    cols.emplace_back(py::cast<std::string>(item.first),
                                      py::cast<std::vector<double>>(item.second));
                }
                return std::make_shared<MarketData>(MarketData::from_columns(std::move(timestamps), cols));
            },
            py::arg("timestamps"), py::arg("columns"),
            "Build MarketData from a timestamp sequence and an ordered {name: values} dict.")
        .def("__len__", &MarketData::size)
        .def("__getitem__", &MarketData::at, py::return_value_policy::reference_internal)
        .def("has_feature", &MarketData::has_feature)
        .def("get_feature_index", &MarketData::get_feature_index)
        .def_property_readonly("feature_map", &MarketData::feature_map);

    py::class_<MarketDataLoader>(m, "MarketDataLoader")
        .def_static("load_csv",
            [](const std::string& path) { return std::make_shared<MarketData>(MarketDataLoader::load_csv(path)); },
            py::arg("path"));

    py::class_<TradingEnv, std::shared_ptr<TradingEnv>>(m, "TradingEnv")
        .def(py::init([](std::shared_ptr<MarketData> data, std::shared_ptr<ExecutionModel> execution,
                         RewardFunctions::RewardFn reward_fn, EnvConfig config) {
                 return std::make_shared<TradingEnv>(std::move(data), std::move(execution),
                                                     std::move(reward_fn), std::move(config));
             }),
             py::arg("data"), py::arg("execution"), py::arg("reward_fn"), py::arg("config") = EnvConfig{})
        .def("reset", [](TradingEnv& env) { return to_numpy(env.reset()); })
        .def("reset", [](TradingEnv& env, std::size_t start, std::size_t end) { return to_numpy(env.reset(start, end)); },
             py::arg("start_index"), py::arg("end_index"))
        .def("step", &TradingEnv::step, py::arg("target_exposure"))
        .def("done", &TradingEnv::done)
        .def("portfolio", &TradingEnv::portfolio, py::return_value_policy::reference_internal)
        .def("current_step", &TradingEnv::current_step)
        .def_property_readonly("episode_start", &TradingEnv::episode_start)
        .def_property_readonly("episode_end", &TradingEnv::episode_end)
        .def_property_readonly("observation_size", &TradingEnv::observation_size)
        .def_property_readonly("market_feature_count", &TradingEnv::market_feature_count)
        .def("observation_names", &TradingEnv::observation_names)
        .def("market_features", [](const TradingEnv& env, std::size_t t) { return to_numpy(env.market_features(t)); },
             py::arg("t"));

    // Functions
    m.def("log_return_reward", &RewardFunctions::log_return_reward);
}
