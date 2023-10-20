#include "benchmark.h"
#include "stopwatch.h"

using namespace Trueface;

void benchmarkFaceLandmarkDetection(const SDKFactory& sdkFactory, BenchmarkParams params) {
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
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < params.numIterations; ++i) {
        tfSdk.detectLargestFace(img, faceBoxAndLandmarks, found);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

    std::cout << "Average time face and landmark detection: " << totalTime / params.numIterations
              << " ms | " << params.numIterations << " iterations" << std::endl;
}