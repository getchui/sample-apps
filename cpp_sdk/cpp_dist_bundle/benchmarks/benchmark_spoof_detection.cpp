#include "benchmark.h"
#include "stopwatch.h"

using namespace Trueface;

void benchmarkSpoofDetection(const SDKFactory& sdkFactory, BenchmarkParams params) {
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
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < params.numIterations; ++i) {
        tfSdk.detectSpoof(img, faceBoxAndLandmarks, label, spoofScore);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

    std::cout << "Average time spoof detection: " << totalTime / params.numIterations
              << " ms | " << params.numIterations << " iterations" << std::endl;
}