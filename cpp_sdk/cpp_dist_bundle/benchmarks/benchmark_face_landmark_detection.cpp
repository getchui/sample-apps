#include "observation.h"
#include "sdkfactory.h"
#include "stopwatch.h"

#include "tf_data_types.h"
#include "tf_sdk.h"

#include <iostream>

using namespace Trueface;

const std::string benchmarkName{"Face and landmark detection"};

void benchmarkFaceLandmarkDetection(const SDKFactory& sdkFactory, BenchmarkParams params, ObservationList& observations) {
    // Initialize the SDK
    auto options = sdkFactory.createBasicConfiguration();
    options.initializeModule.faceDetector = true;

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

    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            errorCode = tfSdk.detectLargestFace(img, faceBoxAndLandmarks, found);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to run face detection" << std::endl;
                return;
            }
        }
    }

    // Time the face detection
    std::vector<float> times;
    times.reserve(params.numIterations);
    for (size_t i = 0; i < params.numIterations; ++i) {
        preciseStopwatch stopwatch;
        tfSdk.detectLargestFace(img, faceBoxAndLandmarks, found);
        times.emplace_back(stopwatch.elapsedTime<float, std::chrono::milliseconds>());
    }

    appendObservationsFromTimes(tfSdk.getVersion(), sdkFactory.isGpuEnabled(),
                                benchmarkName, "", params, times, observations);
}