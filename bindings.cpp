#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "Bar.h"
#include "ExecutionModel.h"
#include "MarketData.h"
#include "MarketDataLoader.h"
#include "Portfolio.h"
#include "RewardFunctions.h"
#include "TradingEnv.h"

namespace py = pybind11;

PYBIND11_MODULE(trading_env_py, m) {
    m.doc() = "C++ Trading Environment Python Bindings";

    // Constants 
    m.attr("DEFAULT_INITIAL_CASH") = DEFAULT_INITIAL_CASH;
    m.attr("DEFAULT_START_INDEX") = DEFAULT_START_INDEX;
    m.attr("DEFAULT_END_INDEX") = DEFAULT_END_INDEX;
    m.attr("SMA_WINDOW_SIZE") = SMA_WINDOW_SIZE;

    m.attr("POSITION_FLAT") = POSITION_FLAT;
    m.attr("OPEN_LONG") = OPEN_LONG;
    m.attr("OPEN_SHORT") = OPEN_SHORT;
    m.attr("MAX_POSITION_CHANGE") = static_cast<double>(OPEN_LONG - OPEN_SHORT);

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
        .def_readwrite("end_index", &EnvConfig::end_index);

    py::class_<Observation>(m, "Observation")
        .def(py::init<>())
        .def_readonly("last_log_return", &Observation::last_log_return)
        .def_readonly("short_window_mean_return", &Observation::short_window_mean_return)
        .def_readonly("current_position", &Observation::current_position)
        .def_readonly("equity", &Observation::equity)
        .def_readonly("cash", &Observation::cash)
        .def_readonly("steps_remaining", &Observation::steps_remaining)
        .def_readonly("forecast", &Observation::forecast);

    py::class_<StepResult>(m, "StepResult")
        .def_readonly("observation", &StepResult::observation)
        .def_readonly("reward", &StepResult::reward)
        .def_readonly("terminated", &StepResult::terminated)
        .def_readonly("truncated", &StepResult::truncated);

    // Core Classes 
    py::class_<ExecutionModel, std::shared_ptr<ExecutionModel>>(m, "ExecutionModel")
        .def(py::init<double, double, double>())
        .def("compute_fill_price", &ExecutionModel::compute_fill_price)
        .def("compute_transaction_cost_log", &ExecutionModel::compute_transaction_cost_log);

    py::class_<Portfolio>(m, "Portfolio")
        .def_property_readonly("position", &Portfolio::position)
        .def_property_readonly("cash", &Portfolio::cash)
        .def_property_readonly("equity", &Portfolio::equity)
        .def_property_readonly("entry_price", &Portfolio::entry_price)
        .def_property_readonly("step_index", &Portfolio::step_index);

    py::class_<MarketData, std::shared_ptr<MarketData>>(m, "MarketData")
        .def("__len__", &MarketData::size)
        .def("__getitem__", &MarketData::at, py::return_value_policy::reference_internal)
        .def("get_feature_index", &MarketData::get_feature_index)
        .def_property_readonly("feature_map", &MarketData::feature_map);

    py::class_<MarketDataLoader>(m, "MarketDataLoader")
        .def_static("load_csv", &MarketDataLoader::load_csv);

    py::class_<TradingEnv, std::shared_ptr<TradingEnv>>(m, "TradingEnv")
        .def(py::init<std::shared_ptr<const MarketData>, std::shared_ptr<ExecutionModel>, RewardFunctions::RewardFn, EnvConfig>())
        .def("reset", &TradingEnv::reset)
        .def("step", &TradingEnv::step)
        .def("done", &TradingEnv::done)
        .def("portfolio", &TradingEnv::portfolio, py::return_value_policy::reference_internal)
        .def("current_step", &TradingEnv::current_step);

    // Functions
    m.def("log_return_reward", &RewardFunctions::log_return_reward);
}
