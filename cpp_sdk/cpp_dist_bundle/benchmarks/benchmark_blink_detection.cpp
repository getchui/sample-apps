#include "memory_high_water_mark.h"
#include "observation.h"
#include "sdkfactory.h"
#include "stopwatch.h"

#include "tf_data_types.h"
#include "tf_sdk.h"

#include <iostream>

using namespace Trueface;
using namespace Trueface::Benchmarks;

const std::string benchmarkName{"Blink detection"};

void benchmarkBlinkDetection(const SDKFactory &sdkFactory, Parameters params,
                             ObservationList &observations) {
    // Baseline memory reading
    auto memoryTracker = MemoryHighWaterMarkTracker();

    // Initialize the SDK
    auto options = sdkFactory.createBasicConfiguration();
    options.initializeModule.faceDetector = true;
    options.initializeModule.blinkDetector = true;

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
    if (errorCode != ErrorCode::NO_ERROR || !found) {
        std::cout << "Unable to get face landmarks in image" << std::endl;
        return;
    }

    std::vector<TFImage> tfImages;
    std::vector<Landmarks> landmarksVec;
    for (size_t i = 0; i < params.batchSize; ++i) {
        tfImages.push_back(img);
        landmarksVec.push_back(landmarks);
    }

    std::vector<BlinkState> blinkstates;

    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            errorCode = tfSdk.detectBlinks(tfImages, landmarksVec, blinkstates);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to run blink detection" << std::endl;
                return;
            }
        }
    }

    // Time the blink detector
    std::vector<float> times;
    times.reserve(params.numIterations);
    for (size_t i = 0; i < params.numIterations; ++i) {
        preciseStopwatch stopwatch;
        tfSdk.detectBlinks(tfImages, landmarksVec, blinkstates);
        times.emplace_back(stopwatch.elapsedTime<float, std::chrono::nanoseconds>());
    }

    observations.emplace_back(tfSdk.getVersion(), sdkFactory.isGpuEnabled(), benchmarkName, "",
                              params, times, memoryTracker.getDifferenceFromBaseline());
}
