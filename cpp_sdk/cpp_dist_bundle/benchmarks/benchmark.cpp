// The following code runs speed benchmarks for the different modules
// The first few inferences are discarded to ensure caching is hot

#include "benchmark.h"
#include "stopwatch.h"
#include "sdkfactory.h"

#include <iostream>
#include <string>
#include <vector>

#include "tf_sdk.h"

using namespace Trueface;

int main() {
    GPUOptions gpuOptions;
    gpuOptions.enableGPU = false; // TODO set this to true to benchmark on GPU
    gpuOptions.deviceIndex = 0;

    GPUModuleOptions gpuModuleOptions;
    gpuModuleOptions.precision = Precision::FP16;

    uint32_t batchSize = 16;
    gpuModuleOptions.maxBatchSize = batchSize;
    gpuModuleOptions.optBatchSize = 1;

    gpuOptions.faceDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceRecognizerGPUOptions = gpuModuleOptions;
    gpuOptions.maskDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.objectDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceLandmarkDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceOrientationDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceBlurDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.spoofDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.blinkDetectorGPUOptions = gpuModuleOptions;

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

    benchmarkPreprocessImage(sdkFactory, {warmup, numWarmup, 0, 200});
    benchmarkFaceImageOrientationDetection(sdkFactory, {warmup, numWarmup, 0, 50*multFactor});
    benchmarkFaceLandmarkDetection(sdkFactory, {warmup, numWarmup, 0, 100*multFactor});
    benchmarkDetailedLandmarkDetection(sdkFactory, {warmup, numWarmup, 0, 100*multFactor});
    benchmarkHeadOrientation(sdkFactory, {warmup, numWarmup, 0, 500*multFactor});
    benchmarkFaceImageBlurDetection(sdkFactory, {warmup, numWarmup, 0, 200*multFactor});
    benchmarkBlinkDetection(sdkFactory, {warmup, numWarmup, 0, 100*multFactor});
    benchmarkGlassesDetection(sdkFactory, {warmup,numWarmup, 0, 200*multFactor});
    benchmarkSpoofDetection(sdkFactory, {warmup, numWarmup, 0, 100*multFactor});
    benchmarkObjectDetection(sdkFactory, ObjectDetectionModel::FAST, {warmup, numWarmup, 0, 100*multFactor});
    benchmarkObjectDetection(sdkFactory, ObjectDetectionModel::ACCURATE, {warmup, numWarmup, 0, 40*multFactor});

    // Benchmarks with batching.
    // On CPU, should be the same speed as a batch size of 1.
    // On GPU, will increase the throughput.
    BenchmarkParams batchBenchmarkParams{warmup, numWarmup, 1, 40*multFactor};
    for (auto currentBatchSize : std::vector<unsigned int>{1, batchSize}) {
        batchBenchmarkParams.batchSize = currentBatchSize;
        if (!gpuOptions.enableGPU) {
            // Trueface::SDK::getFaceFeatureVectors is not supported by the LITE model.
            benchmarkFaceRecognition(sdkFactory, FacialRecognitionModel::LITE, batchBenchmarkParams);
        }
        benchmarkFaceRecognition(sdkFactory, FacialRecognitionModel::LITE_V2, batchBenchmarkParams);
        benchmarkFaceRecognition(sdkFactory, FacialRecognitionModel::TFV5_2, batchBenchmarkParams);
        benchmarkFaceRecognition(sdkFactory, FacialRecognitionModel::TFV6, batchBenchmarkParams);
        benchmarkFaceRecognition(sdkFactory, FacialRecognitionModel::TFV7, batchBenchmarkParams);
        benchmarkMaskDetection(sdkFactory, batchBenchmarkParams);
    }

    return EXIT_SUCCESS;
}