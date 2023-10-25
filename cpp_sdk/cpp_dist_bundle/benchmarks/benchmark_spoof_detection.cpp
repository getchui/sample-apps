#include "memory_high_water_mark.h"
#include "observation.h"
#include "sdkfactory.h"
#include "stopwatch.h"

#include "tf_data_types.h"
#include "tf_sdk.h"

#include <iostream>

using namespace Trueface;
using namespace Trueface::Benchmarks;

const std::string benchmarkName{"Passive spoof detection"};

void benchmarkSpoofDetection(const SDKFactory& sdkFactory, Parameters params, ObservationList& observations) {
    // baseline memory reading
    auto memoryTracker = MemoryHighWaterMarkTracker();

    // Initialize the SDK
    auto options = sdkFactory.createBasicConfiguration();
    options.initializeModule.faceDetector = true;
    options.initializeModule.passiveSpoof = true;

    auto tfSdk = sdkFactory.createSDK(options);

    // Load the image
    TFImage img;
    ErrorCode errorCode = tfSdk.preprocessImage("../images/real_spoof.jpg", img);
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

    float spoofScore;
    SpoofLabel label;

    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            errorCode = tfSdk.detectSpoof(img, faceBoxAndLandmarks, label, spoofScore);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Spoof function failed" << std::endl;
                std::cout << errorCode << std::endl;
                return;
            }
        }
    }

    // Time the spoof detector
    std::vector<float> times;
    times.reserve(params.numIterations);
    for (size_t i = 0; i < params.numIterations; ++i) {
        preciseStopwatch stopwatch;
        tfSdk.detectSpoof(img, faceBoxAndLandmarks, label, spoofScore);
        times.emplace_back(stopwatch.elapsedTime<float, std::chrono::nanoseconds>());
    }

    observations.emplace_back(tfSdk.getVersion(), sdkFactory.isGpuEnabled(),
                              benchmarkName, "", params, times,
                              memoryTracker.getDifferenceFromBaseline());
}