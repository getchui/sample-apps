#include "benchmark.h"
#include "stopwatch.h"

using namespace Trueface;

void benchmarkBlinkDetection(const SDKFactory& sdkFactory, BenchmarkParams params) {
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

    BlinkState blinkstate;

    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            errorCode = tfSdk.detectBlink(img, faceBoxAndLandmarks, blinkstate);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to run blink detection" << std::endl;
                return;
            }
        }
    }

    // Time the blink detector
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < params.numIterations; ++i) {
        tfSdk.detectBlink(img, faceBoxAndLandmarks, blinkstate);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

    std::cout << "Average time blink detection: " << totalTime / params.numIterations
              << " ms | " << params.numIterations << " iterations" << std::endl;
}