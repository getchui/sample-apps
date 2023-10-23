#include "observation.h"

#include <fstream>
#include <ostream>

Observation::Observation(std::string v, bool gpuEnabled, std::string b, std::string bt,
                         std::string m, BenchmarkParams p, float t)
    : version{v}, isGpuEnabled{gpuEnabled}, benchmark{b}, benchmarkSubType{bt},
      measurement{m}, params{p}, timeInMs{t} {
}


std::ostream& operator<<(std::ostream& out, const Observation& observation) {
    out << observation.version << ","
        << (observation.isGpuEnabled ? "GPU" : "CPU") << ","
        << "\"" << observation.benchmark << "\","
        << "\"" << observation.benchmarkSubType << "\","
        << "\"" << observation.measurement << "\","
        << observation.params.batchSize << ","
        << observation.params.numIterations << ","
        << observation.timeInMs;
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
            << "Batch Size, "
            << "Number of Iterations, "
            << "Time (ms)"
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