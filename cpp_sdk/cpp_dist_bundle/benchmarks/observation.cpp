#include "observation.h"

#include <algorithm>
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
    out << observation.version << ","
        << (observation.isGpuEnabled ? "GPU" : "CPU") << ","
        << "\"" << observation.benchmark << "\","
        << "\"" << observation.benchmarkSubType << "\","
        << "\"" << observation.measurementName << "\","
        << observation.measurementValue << ","
        << observation.params.batchSize << ","
        << observation.params.numIterations;

    return out;
}

ObservationCSVWriter::ObservationCSVWriter(const std::string& path)
    : path_{path}, writeHeaders_{!doesFileExist(path)} {

}

void ObservationCSVWriter::write(const ObservationList &observations) {
    // write observations to a csv
    std::ofstream out{path_, std::ios::app};

    if (writeHeaders_) {
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
    std::ifstream f{path_};
    return f.good();
}

void appendObservationsFromTimes(const std::string &version, bool isGpuEnabled,
                                 const std::string &benchmarkName, const std::string &benchmarkSubType,
                                 const Parameters &params, std::vector<float> times,
                                 ObservationList &observations) {
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
    auto variance = std::accumulate(times.begin(), times.end(), 0.0, variance_func);

    std::cout << "Average time " << benchmarkName;
    if (!benchmarkSubType.empty()) {
        std::cout << " (" << benchmarkSubType << ")";
    }
    std::cout << ": " << mean << " ms | " << params.numIterations << " iterations" << std::endl;

    observations.emplace_back(version, isGpuEnabled, benchmarkName, benchmarkSubType, "Total Time (ms)", params, total);
    observations.emplace_back(version, isGpuEnabled, benchmarkName, benchmarkSubType, "Mean Time (ms)", params, mean);
    observations.emplace_back(version, isGpuEnabled, benchmarkName, benchmarkSubType, "Variance (ms)", params, variance);
    observations.emplace_back(version, isGpuEnabled, benchmarkName, benchmarkSubType, "Low (ms)", params, *minmax.first);
    observations.emplace_back(version, isGpuEnabled, benchmarkName, benchmarkSubType, "High (ms)", params, *minmax.second);
}

} // namespace Benchmarks
} // namespace Trueface