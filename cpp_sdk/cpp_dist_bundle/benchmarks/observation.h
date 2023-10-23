#pragma once

#include <iosfwd>
#include <string>
#include <vector>

struct BenchmarkParams {
    bool doWarmup;
    int numWarmup;
    unsigned int batchSize;
    unsigned int numIterations;
};

struct Observation {
    Observation(std::string v, bool gpuEnabled, std::string b, std::string bt,
                std::string m, BenchmarkParams p, float t);

    std::string version;
    bool isGpuEnabled;
    std::string benchmark;
    std::string benchmarkSubType;
    std::string measurement;
    BenchmarkParams params;
    float timeInMs;
};

using ObservationList = std::vector<Observation>;

std::ostream& operator<<(std::ostream& out, const Observation& observation);

class ObservationCSVWriter {
public:
    ObservationCSVWriter(const std::string& path);

    void write(const ObservationList&);

private:
    bool doesFileExist(const std::string& path);
    const std::string& path_;
    bool writeHeaders_;
};