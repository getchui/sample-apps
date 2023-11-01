#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace Trueface {
namespace Benchmarks {

struct Parameters {
    bool doWarmup;
    int numWarmup;
    unsigned int batchSize;
    unsigned int numIterations;
};

struct TimeResult {
    float total;
    float mean;
    float variance;
    float low;
    float high;
};

class Observation {
public:
    Observation(const std::string &version, bool isGpuEnabled, const std::string &benchmarkName,
                const std::string &benchmarkSubType, const Parameters &params,
                const std::vector<float> &times, float memoryUsage);

    const std::string &getVersion() const { return m_version; }
    bool getIsGpuEnabled() const { return m_isGpuEnabled; }
    const std::string &getBenchmarkName() const { return m_benchmarkName; }
    const std::string &getBenchmarkSubType() const { return m_benchmarkSubType; }
    const Parameters &getParameters() const { return m_params; }
    const TimeResult &getTimeResult() const { return m_time; }
    float getMemoryUsage() const { return m_memoryUsage; }

private:
    void emitToUser();

    std::string m_version;
    bool m_isGpuEnabled;
    std::string m_benchmarkName;
    std::string m_benchmarkSubType;
    Parameters m_params;
    TimeResult m_time;
    float m_memoryUsage;
};

std::ostream &operator<<(std::ostream &, const Observation &);

using ObservationList = std::vector<Observation>;

class ObservationCSVWriter {
public:
    ObservationCSVWriter(const std::string &path);

    void write(const ObservationList &);

private:
    bool doesFileExist(const std::string &path);
    const std::string &m_path;
    bool m_writeHeaders;
};

} // namespace Benchmarks
} // namespace Trueface
