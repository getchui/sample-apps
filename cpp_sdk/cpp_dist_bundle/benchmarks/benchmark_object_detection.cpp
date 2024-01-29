#include "memory_high_water_mark.h"
#include "observation.h"
#include "sdkfactory.h"
#include "stopwatch.h"

#include "tf_data_types.h"
#include "tf_sdk.h"

#include <iostream>

using namespace Trueface;
using namespace Trueface::Benchmarks;

const std::string benchmarkName{"Object detection"};

void benchmarkObjectDetection(const SDKFactory &sdkFactory, ObjectDetectionModel objModel,
                              Parameters params, ObservationList &observations) {
    // Baseline memory reading
    auto memoryTracker = MemoryHighWaterMarkTracker();

    // Initialize the SDK with the fast object detection model
    auto options = sdkFactory.createBasicConfiguration();
    options.objModel = objModel;
    options.initializeModule.objectDetector = true;

    auto tfSdk = sdkFactory.createSDK(options);

    // Load the image
    TFImage img;
    ErrorCode errorCode = tfSdk.preprocessImage("../images/bike.jpg", img);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: could not load the image" << std::endl;
        return;
    }

    std::vector<BoundingBox> boundingBoxes;

    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            errorCode = tfSdk.detectObjects(img, boundingBoxes);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Could not run object detection!" << std::endl;
                return;
            }
        }
    }

    // Time the object detector
    std::vector<float> times;
    times.reserve(params.numIterations);
    for (size_t i = 0; i < params.numIterations; ++i) {
        preciseStopwatch stopwatch;
        tfSdk.detectObjects(img, boundingBoxes);
        times.emplace_back(stopwatch.elapsedTime<float, std::chrono::nanoseconds>());
    }

    const std::string mode = (options.objModel == ObjectDetectionModel::FAST) ? "fast" : "accurate";
    observations.emplace_back(tfSdk.getVersion(), sdkFactory.isGpuEnabled(), benchmarkName, mode,
                              params, times, memoryTracker.getDifferenceFromBaseline());
}