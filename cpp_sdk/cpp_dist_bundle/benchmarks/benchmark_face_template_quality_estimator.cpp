#include "memory_high_water_mark.h"
#include "observation.h"
#include "sdkfactory.h"
#include "stopwatch.h"

#include "tf_data_types.h"
#include "tf_sdk.h"

#include <iostream>

using namespace Trueface;
using namespace Trueface::Benchmarks;

const std::string benchmarkName{"Face template quality estimation"};

void benchmarkFaceTemplateQualityEstimation(const SDKFactory &sdkFactory, Parameters params,
                            ObservationList &observations) {
    // baseline memory reading
    auto memoryTracker = MemoryHighWaterMarkTracker();

    // Initialize the SDK
    auto options = sdkFactory.createBasicConfiguration();
    options.initializeModule.faceTemplateQualityEstimator = true;

    auto tfSdk = sdkFactory.createSDK(options);

    // Load the image
    TFImage img;
    ErrorCode errorCode = tfSdk.preprocessImage("../images/headshot.jpg", img);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: could not load the image" << std::endl;
        return;
    }

    FaceBoxAndLandmarks faceBoxAndLandmarks;
    bool found = false;
    errorCode = tfSdk.detectLargestFace(img, faceBoxAndLandmarks, found);

    if (errorCode != ErrorCode::NO_ERROR || !found) {
        std::cout << "Unable to detect face in image" << std::endl;
        return;
    }

    TFFacechip facechip;
    errorCode = tfSdk.extractAlignedFace(img, faceBoxAndLandmarks, facechip);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to extract aligned face for face template quality estimation" << std::endl;
        return;
    }

    std::vector<TFFacechip> facechips;
    for (size_t i = 0; i < params.batchSize; ++i) {
        facechips.push_back(facechip);
    }

    std::vector<bool> areQualitiesGood;
    std::vector<float> scores;

    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            errorCode = tfSdk.estimateFaceTemplateQualities(facechips, areQualitiesGood, scores);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to run face template quality estimation" << std::endl;
                return;
            }
        }
    }

    // Time the mask detector
    std::vector<float> times;
    times.reserve(params.numIterations);
    for (size_t i = 0; i < params.numIterations; ++i) {
        preciseStopwatch stopwatch;
        tfSdk.estimateFaceTemplateQualities(facechips, areQualitiesGood, scores);
        times.emplace_back(stopwatch.elapsedTime<float, std::chrono::nanoseconds>());
    }

    observations.emplace_back(tfSdk.getVersion(), sdkFactory.isGpuEnabled(), benchmarkName, "",
                              params, times, memoryTracker.getDifferenceFromBaseline());
}