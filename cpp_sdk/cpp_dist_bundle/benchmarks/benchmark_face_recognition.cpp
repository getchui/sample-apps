#include "benchmark.h"
#include "stopwatch.h"

using namespace Trueface;

std::string getModelName(FacialRecognitionModel model) {
    if (model == FacialRecognitionModel::TFV5_2) {
        return "TFV5_2";
    } else if (model == FacialRecognitionModel::TFV6) {
        return "TFV6";
    } else if (model == FacialRecognitionModel::TFV7) {
        return "TFV7";
    } else if (model == FacialRecognitionModel::LITE) {
        return "LITE";
    } else if (model == FacialRecognitionModel::LITE_V2) {
        return "LITE V2";
    } else {
        throw std::runtime_error("The model is currently not supported by the benchmarking script");
    }
}

void benchmarkFaceRecognition(FacialRecognitionModel model, const SDKFactory& sdkFactory, BenchmarkParams params) {
    // Initialize the SDK
    auto options = sdkFactory.createBasicConfiguration();
    options.frModel = model;
    options.initializeModule.faceRecognizer = true;

    auto tfSdk = sdkFactory.createSDK(options);

    // Load the image
    TFImage img;
    ErrorCode errorCode = tfSdk.preprocessImage("../images/headshot.jpg", img);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: could not load the image" << std::endl;
        return;
    }

    // Obtain the aligned chip
    FaceBoxAndLandmarks faceBoxAndLandmarks;
    bool found;
    errorCode = tfSdk.detectLargestFace(img, faceBoxAndLandmarks, found);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to detect face when benchmarking face recognition model" << std::endl;
        return;
    }

    TFFacechip facechip;
    errorCode = tfSdk.extractAlignedFace(img, faceBoxAndLandmarks, facechip);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to extract aligned face when benchmarking face recognition model" << std::endl;
        return;
    }

    std::vector<TFFacechip> facechips;
    for (size_t i = 0; i < params.batchSize; ++i) {
        facechips.push_back(facechip);
    }

    std::vector<Faceprint> faceprints;

    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            errorCode = tfSdk.getFaceFeatureVectors(facechips, faceprints);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to run face recognition" << std::endl;
                return;
            }
        }
    }

    preciseStopwatch stopwatch;
    for (size_t i = 0; i < params.numIterations; ++i) {
        tfSdk.getFaceFeatureVectors(facechips, faceprints);
    }

    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

    std::cout << "Average time face recognition " << getModelName(model) << ": " << totalTime / params.numIterations / static_cast<float>(params.batchSize)
              << " ms | batch size = " << params.batchSize << " | " << params.numIterations << " iterations" << std::endl;
}