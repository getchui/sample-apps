// The following code runs speed benchmarks for the different modules
// The first few inferences are discarded to ensure caching is hot

#include "benchmark.h"
#include "stopwatch.h"
#include "sdkfactory.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "tf_sdk.h"

using namespace Trueface;

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

int main() {
    uint32_t batchSize = 16;
    GPUOptions gpuOptions = SDKFactory::createGPUOptions(
        false,        // enableGPU,  NOTE: set this to true to benchmark on GPU
        0,            // deviceIndex
        batchSize,    // maxBatchSize
        1             // optBatchSize
    );

    std::cout << "==========================" << std::endl;
    std::cout << "==========================" << std::endl;
    if (gpuOptions.enableGPU) {
        std::cout << "Using GPU for inference" << std::endl;
    } else {
        std::cout << "Using CPU for inference" << std::endl;
    }
    std::cout << "==========================" << std::endl;
    std::cout << "==========================" << std::endl;

    unsigned int multFactor = 1;
    if (gpuOptions.enableGPU) {
        multFactor = 10;
    }

    bool warmup = true; // Warmup inference to ensure caching is hot
    int numWarmup = 10;
    SDKFactory sdkFactory(gpuOptions);
    ObservationList observations;

    benchmarkPreprocessImage(sdkFactory, {warmup, numWarmup, 0, 200}, observations);
    benchmarkFaceImageOrientationDetection(sdkFactory, {warmup, numWarmup, 0, 50*multFactor}, observations);
    benchmarkFaceLandmarkDetection(sdkFactory, {warmup, numWarmup, 0, 100*multFactor}, observations);
    benchmarkDetailedLandmarkDetection(sdkFactory, {warmup, numWarmup, 0, 100*multFactor}, observations);
    benchmarkHeadOrientation(sdkFactory, {warmup, numWarmup, 0, 500*multFactor}, observations);
    benchmarkFaceImageBlurDetection(sdkFactory, {warmup, numWarmup, 0, 200*multFactor}, observations);
    benchmarkBlinkDetection(sdkFactory, {warmup, numWarmup, 0, 100*multFactor}, observations);
    benchmarkGlassesDetection(sdkFactory, {warmup,numWarmup, 0, 200*multFactor}, observations);
    benchmarkSpoofDetection(sdkFactory, {warmup, numWarmup, 0, 100*multFactor}, observations);
    benchmarkObjectDetection(sdkFactory, ObjectDetectionModel::FAST, {warmup, numWarmup, 0, 100*multFactor}, observations);
    benchmarkObjectDetection(sdkFactory, ObjectDetectionModel::ACCURATE, {warmup, numWarmup, 0, 40*multFactor}, observations);

    // Benchmarks with batching.
    // On CPU, should be the same speed as a batch size of 1.
    // On GPU, will increase the throughput.
    BenchmarkParams batchBenchmarkParams{warmup, numWarmup, 1, 40*multFactor};
    for (auto currentBatchSize : std::vector<unsigned int>{1, batchSize}) {
        batchBenchmarkParams.batchSize = currentBatchSize;
        if (!gpuOptions.enableGPU) {
            // Trueface::SDK::getFaceFeatureVectors is not supported by the LITE model.
            benchmarkFaceRecognition(sdkFactory, FacialRecognitionModel::LITE, batchBenchmarkParams, observations);
        }
        benchmarkFaceRecognition(sdkFactory, FacialRecognitionModel::LITE_V2, batchBenchmarkParams, observations);
        benchmarkFaceRecognition(sdkFactory, FacialRecognitionModel::TFV5_2, batchBenchmarkParams, observations);
        benchmarkFaceRecognition(sdkFactory, FacialRecognitionModel::TFV6, batchBenchmarkParams, observations);
        benchmarkFaceRecognition(sdkFactory, FacialRecognitionModel::TFV7, batchBenchmarkParams, observations);
        benchmarkMaskDetection(sdkFactory, batchBenchmarkParams, observations);
    }

    // write observations to a csv
    std::ofstream out{"benchmarks.csv", std::ios::app};
    if (!out.good()) {
        return EXIT_FAILURE;
    }

    for (const auto& observation : observations) {
        out << observation << "\n";
    }

    out.close();

    return EXIT_SUCCESS;
}