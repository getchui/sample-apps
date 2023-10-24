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

struct Observation {
    Observation(std::string v, bool gpuEnabled, std::string b, std::string bt,
                std::string m, Parameters p, float t);

    std::string version;
    bool isGpuEnabled;
    std::string benchmark;
    std::string benchmarkSubType;
    std::string measurementName;
    Parameters params;
    float measurementValue;
};
std::ostream& operator<<(std::ostream& out, const Observation& observation);

using ObservationList = std::vector<Observation>;

void appendObservationsFromTimes(const std::string &version, bool isGpuEnabled,
                                 const std::string &benchmarkName, const std::string &benchmarkSubType,
                                 const Parameters &params, std::vector<float> times,
                                 ObservationList &observations);

class ObservationCSVWriter {
public:
    ObservationCSVWriter(const std::string& path);

    void write(const ObservationList&);

private:
    bool doesFileExist(const std::string& path);
    const std::string& path_;
    bool writeHeaders_;
};

    } // namespace Benchmarks
} // namespace Trueface;
