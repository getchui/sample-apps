#include "observation.h"
#include "sdkfactory.h"
#include "stopwatch.h"

#include "tf_data_types.h"
#include "tf_sdk.h"

#include <iostream>

using namespace Trueface;

const std::string benchmarkName{"Face image orientation detection"};

void benchmarkFaceImageOrientationDetection(const SDKFactory& sdkFactory, BenchmarkParams params, ObservationList& observations) {
    // Initialize the SDK
    auto options = sdkFactory.createBasicConfiguration();
    options.initializeModule.faceOrientationDetector = true;

    auto tfSdk = sdkFactory.createSDK(options);

    // Load the image
    TFImage img;
    ErrorCode errorCode = tfSdk.preprocessImage("../images/headshot.jpg", img);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: could not load the image" << std::endl;
        return;
    }

    RotateFlags flags;

    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            errorCode = tfSdk.getFaceImageRotation(img, flags);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to compute face image orientation" << std::endl;
                return;
            }
        }
    }

    // Time the mask detector
    std::vector<float> times;
    times.reserve(params.numIterations);
    for (size_t i = 0; i < params.numIterations; ++i) {
        preciseStopwatch stopwatch;
        tfSdk.getFaceImageRotation(img, flags);
        times.emplace_back(stopwatch.elapsedTime<float, std::chrono::milliseconds>());
    }

    appendObservationsFromTimes(tfSdk.getVersion(), sdkFactory.isGpuEnabled(),
                                benchmarkName, "", params, times, observations);
}