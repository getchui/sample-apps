#include "memory_high_water_mark.h"
#include "observation.h"
#include "sdkfactory.h"
#include "stopwatch.h"

#include "tf_data_types.h"
#include "tf_sdk.h"

#include <iostream>

using namespace Trueface;
using namespace Trueface::Benchmarks;

const std::string benchmarkName{"106 face landmark detection"};

void benchmarkDetailedLandmarkDetection(const SDKFactory &sdkFactory, Parameters params,
                                        ObservationList &observations) {
    // baseline memory reading
    auto memoryTracker = MemoryHighWaterMarkTracker();

    // Initialize the SDK
    auto options = sdkFactory.createBasicConfiguration();
    options.initializeModule.landmarkDetector = true;

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

    std::vector<FaceBoxAndLandmarks> fbs;
    std::vector<TFImage> tfImages;
    for (size_t i = 0; i < params.batchSize; ++i) {
        fbs.push_back(faceBoxAndLandmarks);
        tfImages.push_back(img);
    }

    std::vector<Landmarks> landmarksVec;
    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            errorCode = tfSdk.getFaceLandmarks(tfImages, fbs, landmarksVec);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to get detailed landmarks" << std::endl;
                return;
            }
        }
    }

    // Time the landmark detection
    std::vector<float> times;
    times.reserve(params.numIterations);
    for (size_t i = 0; i < params.numIterations; ++i) {
        preciseStopwatch stopwatch;
        tfSdk.getFaceLandmarks(tfImages, fbs, landmarksVec);
        times.emplace_back(stopwatch.elapsedTime<float, std::chrono::nanoseconds>());
    }

    observations.emplace_back(tfSdk.getVersion(), sdkFactory.isGpuEnabled(), benchmarkName, "",
                              params, times, memoryTracker.getDifferenceFromBaseline());
}
