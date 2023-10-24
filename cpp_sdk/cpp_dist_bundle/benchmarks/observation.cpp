#include "observation.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <numeric>
#include <ostream>
#include <sstream>

namespace Trueface {
namespace Benchmarks {

Observation::Observation(std::string v, bool gpuEnabled, std::string b, std::string bt,
                         std::string m, Parameters p, float t)
    : version{v}, isGpuEnabled{gpuEnabled}, benchmark{b}, benchmarkSubType{bt},
      measurementName{m}, params{p}, measurementValue{t} {
}

std::ostream& operator<<(std::ostream& out, const Observation& observation) {
    auto precision{out.precision()};
    out << observation.version << ","
        << (observation.isGpuEnabled ? "GPU" : "CPU") << ","
        << "\"" << observation.benchmark << "\","
        << "\"" << observation.benchmarkSubType << "\","
        << "\"" << observation.measurementName << "\","
        << std::fixed << std::setprecision(3)
        << observation.measurementValue << ","
        << std::defaultfloat << std::setprecision(precision)
        << observation.params.batchSize << ","
        << observation.params.numIterations;

    return out;
}

ObservationCSVWriter::ObservationCSVWriter(const std::string& path)
    : m_path{path}, m_writeHeaders{!doesFileExist(path)} {

}

void ObservationCSVWriter::write(const ObservationList &observations) {
    // write observations to a csv
    std::ofstream out{m_path, std::ios::app};

    if (m_writeHeaders) {
        out << "SDK Version, "
            << "GPU or CPU, "
            << "Benchmark Name, "
            << "Benchmark Type or Model, "
            << "Measurement Taken, "
            << "Measured Value, "
            << "Batch Size, "
            << "Number of Iterations"
            << "\n";
    }

    for (const auto& observation : observations) {
        out << observation << "\n";
    }

    out.close();
}

bool ObservationCSVWriter::doesFileExist(const std::string& path) {
    std::ifstream f{m_path};
    return f.good();
}

void appendObservationsFromTimes(const std::string &version, bool isGpuEnabled,
                                 const std::string &benchmarkName, const std::string &benchmarkSubType,
                                 const Parameters &params, std::vector<float> times,
                                 ObservationList &observations) {
    //
    // times should be passed in NANOSECONDS as milliseconds do not have the resolution for
    // some individual executions. So, the following divide through by 1000 to convert to ms
    // for consumption down stream AFTER the calculations are done.
    //
    std::transform(times.begin(), times.end(), times.begin(), [&params](float val) {
        return val / static_cast<float>(params.batchSize);
    });
    auto total = std::accumulate(times.begin(), times.end(), 0.0);
    auto mean = total / params.numIterations;
    const auto minmax = std::minmax_element(times.begin(), times.end());

    const size_t sz = times.size();
    auto variance_func = [&mean, &sz](float accumulator, const float &val) {
        return accumulator + ((val - mean) * (val - mean) / (sz - 1));
    };
    auto variance = std::accumulate(times.begin(), times.end(), 0.0, variance_func) / 1000.f;

    constexpr float nsPerMs{1000.f * 1000.f};
    total /= nsPerMs;
    mean /= nsPerMs;
    variance /= nsPerMs;
    *minmax.first /= nsPerMs;
    *minmax.second /= nsPerMs;

    // screen output for reporting progress to user
    std::cout << "Average time " << benchmarkName;
    if (!benchmarkSubType.empty()) {
        std::cout << " (" << benchmarkSubType << ")";
    }
    auto precision{std::cout.precision()};
    std::cout << ": " << std::fixed << std::setprecision(3) << mean << " ms | "
        << std::defaultfloat << std::setprecision(precision)
        << params.numIterations << " iterations " << std::endl;

    observations.emplace_back(version, isGpuEnabled, benchmarkName, benchmarkSubType, "Total Time (ms)", params, total);
    observations.emplace_back(version, isGpuEnabled, benchmarkName, benchmarkSubType, "Mean Time (ms)", params, mean);
    observations.emplace_back(version, isGpuEnabled, benchmarkName, benchmarkSubType, "Variance (ms)", params, variance);
    observations.emplace_back(version, isGpuEnabled, benchmarkName, benchmarkSubType, "Low (ms)", params, *minmax.first);
    observations.emplace_back(version, isGpuEnabled, benchmarkName, benchmarkSubType, "High (ms)", params, *minmax.second);
}

} // namespace Benchmarks
} // namespace Trueface