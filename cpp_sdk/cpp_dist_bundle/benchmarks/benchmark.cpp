// The following code runs speed benchmarks for the different modules
// The first few inferences are discarded to ensure caching is hot

#include "benchmark.h"
#include "stopwatch.h"
#include "sdkfactory.h"

#include <iostream>
#include <string>
#include <vector>

#include "tf_sdk.h"

using namespace Trueface;

int main() {
    GPUOptions gpuOptions;
    gpuOptions.enableGPU = false; // TODO set this to true to benchmark on GPU
    gpuOptions.deviceIndex = 0;

    GPUModuleOptions gpuModuleOptions;
    gpuModuleOptions.precision = Precision::FP16;

    uint32_t batchSize = 16;
    gpuModuleOptions.maxBatchSize = batchSize;
    gpuModuleOptions.optBatchSize = 1;

    gpuOptions.faceDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceRecognizerGPUOptions = gpuModuleOptions;
    gpuOptions.maskDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.objectDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceLandmarkDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceOrientationDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceBlurDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.spoofDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.blinkDetectorGPUOptions = gpuModuleOptions;

    std::cout << "==========================" << std::endl;
    std::cout << "==========================" << std::endl;
    if (gpuOptions.enableGPU) {
        std::cout << "Using GPU for inference" << std::endl;
    } else {
        std::cout << "Using CPU for inference" << std::endl;
    }
    std::cout << "==========================" << std::endl;
    std::cout << "==========================" << std::endl;

    unsigned int multFactor = 1;
    if (gpuOptions.enableGPU) {
        multFactor = 10;
    }

    bool warmup = true; // Warmup inference to ensure caching is hot
    int numWarmup = 10;
    SDKFactory sdkFactory(gpuOptions);

    benchmarkPreprocessImage(sdkFactory, {warmup, numWarmup, 0, 200});
    benchmarkFaceImageOrientationDetection(sdkFactory, {warmup, numWarmup, 0, 50*multFactor});
    benchmarkFaceLandmarkDetection(sdkFactory, {warmup, numWarmup, 0, 100*multFactor});
    benchmarkDetailedLandmarkDetection(sdkFactory, {warmup, numWarmup, 0, 100*multFactor});
    benchmarkHeadOrientation(sdkFactory, {warmup, numWarmup, 0, 500*multFactor});
    benchmarkFaceImageBlurDetection(sdkFactory, {warmup, numWarmup, 0, 200*multFactor});
    benchmarkBlinkDetection(sdkFactory, {warmup, numWarmup, 0, 100*multFactor});
    benchmarkGlassesDetection(sdkFactory, {warmup,numWarmup, 0, 200*multFactor});
    benchmarkSpoofDetection(sdkFactory, {warmup, numWarmup, 0, 100*multFactor});
    benchmarkObjectDetection(sdkFactory, ObjectDetectionModel::FAST, {warmup, numWarmup, 0, 100*multFactor});
    benchmarkObjectDetection(sdkFactory, ObjectDetectionModel::ACCURATE, {warmup, numWarmup, 0, 40*multFactor});

    // Benchmarks with batching.
    // On CPU, should be the same speed as a batch size of 1.
    // On GPU, will increase the throughput.
    BenchmarkParams batchBenchmarkParams{warmup, numWarmup, 1, 40*multFactor};
    for (auto currentBatchSize : std::vector<unsigned int>{1, batchSize}) {
        batchBenchmarkParams.batchSize = currentBatchSize;
        if (!gpuOptions.enableGPU) {
            // Trueface::SDK::getFaceFeatureVectors is not supported by the LITE model.
            benchmarkFaceRecognition(FacialRecognitionModel::LITE, sdkFactory, batchBenchmarkParams);
        }
        benchmarkFaceRecognition(FacialRecognitionModel::LITE_V2, sdkFactory, batchBenchmarkParams);
        benchmarkFaceRecognition(FacialRecognitionModel::TFV5_2, sdkFactory, batchBenchmarkParams);
        benchmarkFaceRecognition(FacialRecognitionModel::TFV6, sdkFactory, batchBenchmarkParams);
        benchmarkFaceRecognition(FacialRecognitionModel::TFV7, sdkFactory, batchBenchmarkParams);
        benchmarkMaskDetection(sdkFactory, batchBenchmarkParams);
    }

    return 0;
}

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

void benchmarkDetailedLandmarkDetection(const SDKFactory& sdkFactory, BenchmarkParams params) {
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

    Landmarks landmarks;

    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            tfSdk.getFaceLandmarks(img, faceBoxAndLandmarks, landmarks);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to get detailed landmarks" << std::endl;
                return;
            }
        }
    }

    // Time the landmark detection
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < params.numIterations; ++i) {
        tfSdk.getFaceLandmarks(img, faceBoxAndLandmarks, landmarks);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

    std::cout << "Average time 106 face landmark detection: " << totalTime / params.numIterations
              << " ms | " << params.numIterations << " iterations" << std::endl;


}

void benchmarkHeadOrientation(const SDKFactory& sdkFactory, BenchmarkParams params) {
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

    std::cout << "Average time head orientation: " << totalTime / params.numIterations
              << " ms | " << params.numIterations << " iterations" << std::endl;


}

void benchmarkFaceImageOrientationDetection(const SDKFactory& sdkFactory, BenchmarkParams params) {
    // Initialize the SDK
    auto options = sdkFactory.createBasicConfiguration();
    options.initializeModule.faceOrientationDetector = true;

    auto tfSdk = sdkFactory.createSDK(options);

    // Load the image
    TFImage img;
    ErrorCode errorCode = tfSdk.preprocessImage("../images/headshot.jpg", img);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: could not load the image" << std::endl;
        return;
    }

    RotateFlags flags;

    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            errorCode = tfSdk.getFaceImageRotation(img, flags);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to compute face image orientation" << std::endl;
                return;
            }
        }
    }

    // Time the mask detector
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < params.numIterations; ++i) {
        tfSdk.getFaceImageRotation(img, flags);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

    std::cout << "Average time face image orientation detection: " << totalTime / params.numIterations
              << " ms  | " << params.numIterations << " iterations" << std::endl;

}

void benchmarkFaceImageBlurDetection(const SDKFactory& sdkFactory, BenchmarkParams params) {
    // Initialize the SDK
    auto options = sdkFactory.createBasicConfiguration();
    options.initializeModule.faceBlurDetector = true;

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

    TFFacechip facechip;
    errorCode = tfSdk.extractAlignedFace(img, faceBoxAndLandmarks, facechip);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to extract aligned face for mask detection" << std::endl;
        return;
    }

    FaceImageQuality quality;
    float score{};

    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            errorCode = tfSdk.detectFaceImageBlur(facechip, quality, score);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to detect face image blur" << std::endl;
                return;
            }
        }
    }

    // Time the mask detector
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < params.numIterations; ++i) {
        tfSdk.detectFaceImageBlur(facechip, quality, score);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

    std::cout << "Average time face image blur detection: " << totalTime / params.numIterations
              << " ms  | " << params.numIterations << " iterations" << std::endl;

}

void benchmarkGlassesDetection(const SDKFactory& sdkFactory, BenchmarkParams params) {
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

    std::cout << "Average time glasses detection: " << totalTime / params.numIterations
              << " ms | " << params.numIterations << " iterations" << std::endl;

}

void benchmarkMaskDetection(const SDKFactory& sdkFactory, BenchmarkParams params) {
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

    TFFacechip facechip;
    errorCode = tfSdk.extractAlignedFace(img, faceBoxAndLandmarks, facechip);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to extract aligned face for mask detection" << std::endl;
        return;
    }

    std::vector<TFFacechip> facechips;
    for (size_t i = 0; i < params.batchSize; ++i) {
        facechips.push_back(facechip);
    }

    std::vector<MaskLabel> maskLabels;
    std::vector<float> maskScores;

    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            errorCode = tfSdk.detectMasks(facechips, maskLabels, maskScores);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to run mask detection" << std::endl;
                return;
            }
        }
    }

    // Time the mask detector
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < params.numIterations; ++i) {
        tfSdk.detectMasks(facechips, maskLabels, maskScores);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

    std::cout << "Average time mask detection: " << totalTime / params.numIterations / static_cast<float>(params.batchSize)
              << " ms | batch size = " << params.batchSize << " | " << params.numIterations << " iterations" << std::endl;

}

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
