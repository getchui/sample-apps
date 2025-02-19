// The following code runs speed benchmarks for the different modules
// The first few inferences are discarded to ensure caching is hot

#include "benchmark.h"
#include "observation.h"
#include "sdkfactory.h"
#include "stopwatch.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "tf_sdk.h"

using namespace Trueface;

int main() {
    uint32_t batchSize = 16;
    GPUOptions gpuOptions = Benchmarks::SDKFactory::createGPUOptions(
        false,     // enableGPU,  NOTE: set this to true to benchmark on GPU
        0,         // deviceIndex
        batchSize, // maxBatchSize
        1          // optBatchSize
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
    Benchmarks::SDKFactory sdkFactory(gpuOptions);
    Benchmarks::ObservationList observations;

    benchmarkPreprocessImage(sdkFactory, {warmup, numWarmup, 1, 200}, observations);

    benchmarkFaceLandmarkDetection(sdkFactory, {warmup, numWarmup, 1, 100 * multFactor}, observations);

    benchmarkHeadOrientation(sdkFactory, {warmup, numWarmup, 1, 500 * multFactor}, observations);

    benchmarkGlassesDetection(sdkFactory, {warmup, numWarmup, 1, 200 * multFactor}, observations);

    benchmarkSpoofDetection(sdkFactory, {warmup, numWarmup, 1, 100 * multFactor}, observations);

    benchmarkObjectDetection(sdkFactory, ObjectDetectionModel::FAST, {warmup, numWarmup, 1, 100 * multFactor}, observations);

    benchmarkObjectDetection(sdkFactory, ObjectDetectionModel::ACCURATE, {warmup, numWarmup, 1, 40 * multFactor}, observations);

    Benchmarks::Parameters frBenchmarkParams{warmup, numWarmup, 1, 40 * multFactor};
    std::vector<unsigned int> batchSizes;
    if (gpuOptions.enableGPU) {
        // Only test with batching when GPU is enabled.
        // Batching is not supported by CPU and will not cause a speedup.
        batchSizes = {1, batchSize};
    } else {
        batchSizes = {1};
    }
    // Methods which support batch inference
    for (auto currentBatchSize : batchSizes) {
        frBenchmarkParams.batchSize = currentBatchSize;
        benchmarkFaceImageOrientationDetection(sdkFactory, {warmup, numWarmup, currentBatchSize, 50 * multFactor}, observations);

        benchmarkDetailedLandmarkDetection(sdkFactory, {warmup, numWarmup, currentBatchSize, 100 * multFactor}, observations);

        benchmarkFaceImageBlurDetection(sdkFactory, {warmup, numWarmup, currentBatchSize, 200 * multFactor}, observations);

        benchmarkBlinkDetection(sdkFactory, {warmup, numWarmup, currentBatchSize, 100 * multFactor}, observations);

        benchmarkMaskDetection(sdkFactory, {warmup, numWarmup, currentBatchSize, 200 * multFactor}, observations);

        benchmarkFaceTemplateQualityEstimation(sdkFactory,  {warmup, numWarmup, currentBatchSize, 200 * multFactor}, observations);

        benchmarkFaceRecognition(sdkFactory, FacialRecognitionModel::LITE_V2, frBenchmarkParams, observations);

        benchmarkFaceRecognition(sdkFactory, FacialRecognitionModel::LITE_V3, frBenchmarkParams, observations);

        benchmarkFaceRecognition(sdkFactory, FacialRecognitionModel::TFV5_2, frBenchmarkParams, observations);

        benchmarkFaceRecognition(sdkFactory, FacialRecognitionModel::TFV6, frBenchmarkParams, observations);

        benchmarkFaceRecognition(sdkFactory, FacialRecognitionModel::TFV7, frBenchmarkParams, observations);
    }

    Benchmarks::ObservationCSVWriter csv{"benchmarks.csv"};
    csv.write(observations);

    return EXIT_SUCCESS;
}