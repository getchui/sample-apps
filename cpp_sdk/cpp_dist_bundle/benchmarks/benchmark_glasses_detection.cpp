#include "benchmark.h"
#include "stopwatch.h"

#include "tf_data_types.h"
#include "tf_sdk.h"

#include <iostream>

using namespace Trueface;

const std::string benchmarkName{"Eyeglasses detection"};

void benchmarkGlassesDetection(const SDKFactory& sdkFactory, BenchmarkParams params, ObservationList& observations) {
    // Initialize the SDK
    auto options = sdkFactory.createBasicConfiguration();
    options.initializeModule.eyeglassDetector = true;

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

    GlassesLabel label;
    float score;

    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            errorCode = tfSdk.detectGlasses(img, faceBoxAndLandmarks, label, score);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to run glasses detection" << std::endl;
                return;
            }
        }
    }

    // Time the mask detector
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < params.numIterations; ++i) {
        tfSdk.detectGlasses(img, faceBoxAndLandmarks, label, score);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    auto avgTime = totalTime / params.numIterations;

    std::cout << "Average time glasses detection: " << avgTime
              << " ms | " << params.numIterations << " iterations" << std::endl;

    observations.emplace_back(tfSdk.getVersion(), sdkFactory.isGpuEnabled(), benchmarkName, "", "Average Time", params, avgTime);
}