#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>

#include <gtest/gtest.h>

#include "Exceptions.h"
#include "MarketDataLoader.h"

namespace {

std::string write_csv(const std::string& name, const std::string& content) {
    const auto path = std::filesystem::path(TEST_TMP_DIR) / name;
    std::ofstream(path, std::ios::binary) << content;
    return path.string();
}

const char kHeader[] = "timestamp,open,close,forecast\n";

}  // namespace

TEST(Loader, LoadsValidFileWithCrlfAndBlankLines) {
    auto path = write_csv("valid.csv",
        "timestamp,open,close,forecast\r\n"
        "1,100.0,101.5,0.25\r\n"
        "2, 101.5 ,102,-1e-3\r\n"
        "\r\n");
    MarketData data = MarketDataLoader::load_csv(path);
    ASSERT_EQ(data.size(), 2u);
    EXPECT_EQ(data[1].timestamp, 2);
    EXPECT_DOUBLE_EQ(data[0].get_feature(data.get_feature_index("close")), 101.5);
    EXPECT_DOUBLE_EQ(data[1].get_feature(data.get_feature_index("forecast")), -1e-3);
    EXPECT_TRUE(data.has_feature("open"));
    EXPECT_FALSE(data.has_feature("timestamp"));
}

struct MalformedCase {
    const char* name;
    std::string content;

    friend void PrintTo(const MalformedCase& c, std::ostream* os) { *os << c.name; }
};

class MalformedCsv : public ::testing::TestWithParam<MalformedCase> {};

TEST_P(MalformedCsv, ThrowsDataLoadError) {
    auto path = write_csv(std::string(GetParam().name) + ".csv", GetParam().content);
    EXPECT_THROW(MarketDataLoader::load_csv(path), DataLoadError);
}

INSTANTIATE_TEST_SUITE_P(Loader, MalformedCsv, ::testing::Values(
    MalformedCase{"empty_file", ""},
    MalformedCase{"header_only", kHeader},
    MalformedCase{"wrong_first_column", "time,open,close\n1,1,1\n"},
    MalformedCase{"no_feature_columns", "timestamp\n1\n"},
    MalformedCase{"duplicate_column", "timestamp,open,close,open\n1,1,1,1\n"},
    MalformedCase{"empty_column_name", "timestamp,open,,close\n1,1,1,1\n"},
    MalformedCase{"trailing_garbage", std::string(kHeader) + "1,100,1.5abc,0\n"},
    MalformedCase{"garbage_timestamp", std::string(kHeader) + "1x,100,100,0\n"},
    MalformedCase{"fractional_timestamp", std::string(kHeader) + "1.5,100,100,0\n"},
    MalformedCase{"empty_field", std::string(kHeader) + "1,100,,0\n"},
    MalformedCase{"too_few_columns", std::string(kHeader) + "1,100,100\n"},
    MalformedCase{"too_many_columns", std::string(kHeader) + "1,100,100,0,7\n"},
    MalformedCase{"trailing_comma", std::string(kHeader) + "1,100,100,0,\n"},
    MalformedCase{"nan_value", std::string(kHeader) + "1,100,nan,0\n"},
    MalformedCase{"inf_value", std::string(kHeader) + "1,100,inf,0\n"},
    MalformedCase{"non_increasing_timestamp", std::string(kHeader) + "2,100,100,0\n2,100,100,0\n"},
    MalformedCase{"decreasing_timestamp", std::string(kHeader) + "2,100,100,0\n1,100,100,0\n"}
), [](const auto& info) { return std::string(info.param.name); });

TEST(Loader, MissingFileThrows) {
    EXPECT_THROW(MarketDataLoader::load_csv("/definitely/not/here.csv"), DataLoadError);
}

TEST(MarketData, FromColumnsValidates) {
    EXPECT_THROW(MarketData::from_columns({1, 2}, {}), DataLoadError);
    EXPECT_THROW(MarketData::from_columns({1, 2}, {{"a", {1.0}}}), DataLoadError);
    EXPECT_THROW(MarketData::from_columns({1, 2}, {{"a", {1.0, 2.0}}, {"a", {1.0, 2.0}}}), DataLoadError);
    EXPECT_THROW(MarketData::from_columns({1, 2}, {{"a", {1.0, std::nan("")}}}), DataLoadError);
    EXPECT_THROW(MarketData::from_columns({2, 1}, {{"a", {1.0, 2.0}}}), DataLoadError);
    EXPECT_THROW(MarketData::from_columns({1, 2}, {{"a", {1.0, 2.0}}}).get_feature_index("b"), ConfigError);
}
