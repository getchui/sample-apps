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

void Observation::emitToUser() {
    // screen output for reporting progress to user
    std::cout << "Average time " << m_benchmarkName;
    if (!m_benchmarkSubType.empty()) {
        std::cout << " (" << m_benchmarkSubType << ")";
    }

    auto precision{std::cout.precision()};
    std::cout << ": " << std::fixed << std::setprecision(3) << m_time.mean << " ms"
        << std::defaultfloat << std::setprecision(precision);

    if (m_params.batchSize > 1) {
        std::cout << " | batch size = " << m_params.batchSize;
    }

    std::cout << " | " << m_params.numIterations << " iterations " << std::endl;
}

TimeResult summarizeTimes(const Parameters &params, std::vector<float> times) {
    //
    // times should be passed in NANOSECONDS as milliseconds do not have the resolution for
    // some individual executions. So, the following divide through by 1000 to convert to ms
    // for consumption down stream AFTER the calculations are done.
    //
    std::transform(times.begin(), times.end(), times.begin(), [&params](float val) {
        return val / static_cast<float>(params.batchSize);
    });
    auto total = std::accumulate(times.begin(), times.end(), 0.0f);
    auto mean = total / params.numIterations;
    const auto minmax = std::minmax_element(times.begin(), times.end());

    const size_t sz = times.size();
    auto variance_func = [&mean, &sz](float accumulator, const float &val) {
        return accumulator + ((val - mean) * (val - mean) / (sz - 1));
    };
    auto variance = std::accumulate(times.begin(), times.end(), 0.0f, variance_func) / 1000.f;

    constexpr float nsPerMs{1000.f * 1000.f};
    return TimeResult{
        total / nsPerMs,
        mean / nsPerMs,
        variance / (nsPerMs * 1000.f),
        *minmax.first / nsPerMs,
        *minmax.second / nsPerMs};
}

Observation::Observation(const std::string &version, bool isGpuEnabled,
                const std::string &benchmarkName, const std::string &benchmarkSubType,
                const Parameters &params, const std::vector<float> &times,
                float memoryUsage)
    : m_version{version}, m_isGpuEnabled{isGpuEnabled}, m_benchmarkName{benchmarkName},
    m_benchmarkSubType{benchmarkSubType}, m_params{params},
    m_time{summarizeTimes(params, times)}, m_memoryUsage{memoryUsage} {
    emitToUser();
}

std::ostream& operator<<(std::ostream& out, const Observation& o) {
    const auto& params = o.getParameters();
    const auto& time = o.getTimeResult();
    auto precision{out.precision()};
    out << o.getVersion() << ", "
        << (o.getIsGpuEnabled() ? "GPU" : "CPU") << ","
        << "\"" << o.getBenchmarkName() << "\","
        << "\"" << o.getBenchmarkSubType() << "\","
        << params.batchSize << ","
        << params.numIterations << ","
        << std::fixed << std::setprecision(3)
        << time.total << "," << time.mean << "," << time.variance << ","
        << time.low << "," << time.high << ","
        << o.getMemoryUsage()
        // reset iomanip
        << std::defaultfloat << std::setprecision(precision);

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
            << "Batch Size, " << "Number of Iterations, "
            << "Total Time (ms), Mean Time (ms), Variance (ms), Low (ms), High (ms), "
            << "Memory Usage (MB)"
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


} // namespace Benchmarks
} // namespace Trueface