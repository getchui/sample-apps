#include "benchmark.h"
#include "stopwatch.h"

using namespace Trueface;

void benchmarkObjectDetection(const SDKFactory& sdkFactory, ObjectDetectionModel objModel, BenchmarkParams params) {
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

    // Time the creation of the feature vector
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < params.numIterations; ++i) {
        tfSdk.detectObjects(img, boundingBoxes);

    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

    const std::string mode = (options.objModel == ObjectDetectionModel::FAST) ? "fast" : "accurate";

    std::cout << "Average time object detection (" + mode + " mode): " << totalTime / params.numIterations
              << " ms | " << params.numIterations << " iterations" << std::endl;
}