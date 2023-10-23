#include "benchmark.h"
#include "stopwatch.h"

#include "tf_data_types.h"
#include "tf_sdk.h"

#include <iostream>

using namespace Trueface;

const std::string benchmarkName{"Head orientation detection (yaw, pitch, roll)"};

void benchmarkHeadOrientation(const SDKFactory& sdkFactory, BenchmarkParams params, ObservationList& observations) {
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
    errorCode = tfSdk.detectLargestFace(img, faceBoxAndLandmarks, found);

    if (errorCode != ErrorCode::NO_ERROR || !found) {
        std::cout << "Unable to detect face in image" << std::endl;
        return;
    }

    Landmarks landmarks;
    errorCode = tfSdk.getFaceLandmarks(img, faceBoxAndLandmarks, landmarks);

    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to detect landmarks" << std::endl;
        return;
    }

    float yaw, pitch, roll;
    std::array<double, 3> rotationVec, translationVec;

    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            errorCode = tfSdk.estimateHeadOrientation(img, faceBoxAndLandmarks, landmarks, yaw, pitch, roll, rotationVec, translationVec);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Unable to run head orientation method" << std::endl;
                return;
            }
        }
    }

    // Time the head orientation
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < params.numIterations; ++i) {
        tfSdk.estimateHeadOrientation(img, faceBoxAndLandmarks, landmarks, yaw, pitch, roll, rotationVec, translationVec);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    auto avgTime = totalTime / params.numIterations;

    std::cout << "Average time head orientation: " << avgTime
              << " ms | " << params.numIterations << " iterations" << std::endl;

    observations.emplace_back(sdkFactory.isGpuEnabled(), benchmarkName, "", "Average Time", params, avgTime);
}