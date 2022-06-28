// The following code runs speed benchmarks for the different modules
// The first inference speed for each module is discarded due to lazy initialization of models

#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib> 

#include "tf_sdk.h"
#include <fstream>

using namespace Trueface;

typedef std::chrono::high_resolution_clock Clock;

void benchmarkFaceRecognition(const std::string& license, FacialRecognitionModel model, const GPUOptions& gpuOptions, unsigned int batchSize = 1, unsigned int numIterations = 100);
void benchmarkObjectDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations = 100);
void benchmarkFaceLandmarkDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations = 100);
void benchmarkDetailedLandmarkDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations = 100);
void benchmarkPreprocessImage(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations = 100);
void benchmarkMaskDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int batchSize = 1, unsigned int numIterations = 100);
void benchmarkBlinkDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations = 100);
void benchmarkSpoofDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations = 100);
void benchmarkHeadOrientation(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations = 200);

int main() {
    const std::string license = TRUEFACE_TOKEN;
    std::cout << "Running speed benchmarks with 1280x720 image\n";

    GPUOptions gpuOptions;
    gpuOptions.enableGPU = true; // TODO set this to true to benchmark on GPU
    gpuOptions.deviceIndex = 0;

    GPUModuleOptions gpuModuleOptions;
    gpuModuleOptions.precision = Precision::FP16;

    int32_t batchSize = 4;
    gpuModuleOptions.maxBatchSize = batchSize;
    gpuModuleOptions.optBatchSize = batchSize;

    gpuOptions.faceDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceRecognizerGPUOptions = gpuModuleOptions;
    gpuOptions.maskDetectorGPUOptions = gpuModuleOptions;

    if (gpuOptions.enableGPU) {
        std::cout << "Using GPU for inference" << std::endl;
    }

    int multFactor = 1;
    if (gpuOptions.enableGPU) {
        multFactor = 10;
    }

    benchmarkPreprocessImage(license, gpuOptions, 200);
    benchmarkFaceLandmarkDetection(license, gpuOptions);
    benchmarkDetailedLandmarkDetection(license, gpuOptions);
    benchmarkBlinkDetection(license, gpuOptions);
    benchmarkSpoofDetection(license, gpuOptions);
    benchmarkMaskDetection(license, gpuOptions);
    benchmarkHeadOrientation(license, gpuOptions);

    if (!gpuOptions.enableGPU) {
        // Trueface::SDK::getFaceFeatureVectors is not supported by the LITE and LITE_V2 models.
        benchmarkFaceRecognition(license, FacialRecognitionModel::LITE, gpuOptions, 1,  200);
        benchmarkFaceRecognition(license, FacialRecognitionModel::LITE_V2, gpuOptions, 1,  200);
    }
    benchmarkFaceRecognition(license, FacialRecognitionModel::TFV6, gpuOptions, 1,  40 * multFactor);
    benchmarkFaceRecognition(license, FacialRecognitionModel::TFV5, gpuOptions, 1,  40 * multFactor);
    benchmarkFaceRecognition(license, FacialRecognitionModel::FULL, gpuOptions, 1,  40 * multFactor);
    benchmarkObjectDetection(license, gpuOptions);

    // Benchmarks with batching.
    // On CPU, should be the same speed as a batch size of 1.
    // On GPU, will increase the throughput.
    benchmarkFaceRecognition(license, FacialRecognitionModel::TFV5, gpuOptions, batchSize, 40 * multFactor);
    benchmarkMaskDetection(license, gpuOptions, batchSize, 40 * multFactor);

    return 0;
}

std::string getModelName(FacialRecognitionModel model) {
    if (model == FacialRecognitionModel::TFV5) {
        return "TFV5";
    } else if (model == FacialRecognitionModel::TFV6) {
        return "TFV6";
    } else if (model == FacialRecognitionModel::FULL) {
        return "FULL";
    } else if (model == FacialRecognitionModel::LITE) {
        return "LITE";
    } else if (model == FacialRecognitionModel::LITE_V2) {
        return "LITE V2";
    } else {
        throw std::runtime_error("The model is currently not supported by the benchmarking script");
    }
}

void benchmarkFaceRecognition(const std::string& license, FacialRecognitionModel model, const GPUOptions& gpuOptions, unsigned int batchSize, unsigned int numIterations) {
    // Initialize the SDK
    ConfigurationOptions options;
    options.modelsPath = "./";
    auto modelsPath = std::getenv("MODELS_PATH");
    if (modelsPath) {
        options.modelsPath = modelsPath;
    }
    options.gpuOptions = gpuOptions;
    options.frModel = model;

    // Since we initialize the module, we do not need to discard the first inference time.
    InitializeModule initializeModule;
    initializeModule.faceRecognizer = true;
    options.initializeModule = initializeModule;

    SDK tfSdk(options);
    bool valid = tfSdk.setLicense(license);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        exit (EXIT_FAILURE);
    }

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
    tfSdk.detectLargestFace(img, faceBoxAndLandmarks, found);

    TFFacechip facechip;
    tfSdk.extractAlignedFace(img, faceBoxAndLandmarks, facechip);

    std::vector<TFFacechip> facechips;
    for (size_t i = 0; i < batchSize; ++i) {
        facechips.push_back(facechip);
    }

    std::vector<Faceprint> faceprints;

    // Time the creation of the feature vector
    auto t1 = Clock::now();
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.getFaceFeatureVectors(facechips, faceprints);
    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Average time face recognition " << getModelName(model) << ": " << totalTime / numIterations / static_cast<float>(batchSize)
              << " ms | batch size = " << batchSize << " | " << numIterations << " iterations" << std::endl;

}

void benchmarkObjectDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations) {
    // Initialize the SDK with the fast object detection model
    ConfigurationOptions options;
    options.modelsPath = "./";
    auto modelsPath = std::getenv("MODELS_PATH");
    if (modelsPath) {
        options.modelsPath = modelsPath;
    }
    options.objModel = ObjectDetectionModel::FAST;

    // Since we initialize the module, we do not need to discard the first inference time.
    InitializeModule initializeModule;
    initializeModule.faceDetector = true;
    options.initializeModule = initializeModule;

    SDK tfSdk(options);
    bool valid = tfSdk.setLicense(license);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        exit (EXIT_FAILURE);
    }

    // Load the image
    TFImage img;
    ErrorCode errorCode = tfSdk.preprocessImage("../images/bike.jpg", img);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: could not load the image" << std::endl;
        return;
    }

    std::vector<BoundingBox> boundingBoxes;

    // Time the creation of the feature vector
    auto t1 = Clock::now();
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.detectObjects(img, boundingBoxes);

    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Average time object detection (fast mode): " << totalTime / numIterations
              << " ms | " << numIterations << " iterations" << std::endl;

}

void benchmarkPreprocessImage(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations) {
    // Initialize the SDK
    ConfigurationOptions options;
    options.modelsPath = "./";
    auto modelsPath = std::getenv("MODELS_PATH");
    if (modelsPath) {
        options.modelsPath = modelsPath;
    }
    options.gpuOptions = gpuOptions;

    SDK tfSdk(options);
    bool valid = tfSdk.setLicense(license);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        exit (EXIT_FAILURE);
    }

    // Read the encoded image into memory
    std::ifstream file("../images/headshot.jpg", std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cout << "Unable to load the image" << std::endl;
        return;
    }

    // Run once to ensure everything works
    TFImage img;
    auto errorcode = tfSdk.preprocessImage(buffer, img);
    if (errorcode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to preprocess the image" << std::endl;
        return;
    }

    // Time the preprocessImage function
    auto t1 = Clock::now();
    for (size_t i = 0; i < numIterations; ++i) {
        TFImage newImg;
        tfSdk.preprocessImage(buffer, newImg);
    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Average time preprocessImage (" << img->getWidth() << "x" << img->getHeight() << "): " <<
    totalTime / numIterations << " ms | " << numIterations << " iterations" << std::endl;
}

void benchmarkDetailedLandmarkDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations) {
    // Initialize the SDK
    ConfigurationOptions options;
    options.modelsPath = "./";
    auto modelsPath = std::getenv("MODELS_PATH");
    if (modelsPath) {
        options.modelsPath = modelsPath;
    }
    options.gpuOptions = gpuOptions;
    options.smallestFaceHeight = 40;

    // Since we initialize the module, we do not need to discard the first inference time.
    InitializeModule initializeModule;
    initializeModule.landmarkDetector = true;
    options.initializeModule = initializeModule;

    SDK tfSdk(options);
    bool valid = tfSdk.setLicense(license);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        exit (EXIT_FAILURE);
    }

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

    // Time the landmark detection
    auto t1 = Clock::now();
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.getFaceLandmarks(img, faceBoxAndLandmarks, landmarks);
    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Average time 106 face landmark detection: " << totalTime / numIterations
              << " ms | " << numIterations << " iterations" << std::endl;


}

void benchmarkHeadOrientation(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations) {
    // Initialize the SDK
    ConfigurationOptions options;
    options.modelsPath = "./";
    auto modelsPath = std::getenv("MODELS_PATH");
    if (modelsPath) {
        options.modelsPath = modelsPath;
    }
    options.gpuOptions = gpuOptions;
    options.smallestFaceHeight = 40;

    // Since we initialize the module, we do not need to discard the first inference time.
    InitializeModule initializeModule;
    initializeModule.faceDetector = true;
    options.initializeModule = initializeModule;

    SDK tfSdk(options);
    bool valid = tfSdk.setLicense(license);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        exit (EXIT_FAILURE);
    }

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

    float yaw, pitch, roll;

    // Time the head orientation
    auto t1 = Clock::now();
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.estimateHeadOrientation(img, faceBoxAndLandmarks, yaw, pitch, roll);
    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Average time head orientation: " << totalTime / numIterations
              << " ms | " << numIterations << " iterations" << std::endl;


}

void benchmarkMaskDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int batchSize, unsigned int numIterations) {
    // Initialize the SDK
    ConfigurationOptions options;
    options.modelsPath = "./";
    auto modelsPath = std::getenv("MODELS_PATH");
    if (modelsPath) {
        options.modelsPath = modelsPath;
    }
    options.gpuOptions = gpuOptions;
    options.smallestFaceHeight = 40;

    // Since we initialize the module, we do not need to discard the first inference time.
    InitializeModule initializeModule;
    initializeModule.faceDetector = true;
    options.initializeModule = initializeModule;

    SDK tfSdk(options);
    bool valid = tfSdk.setLicense(license);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        exit (EXIT_FAILURE);
    }

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
    tfSdk.extractAlignedFace(img, faceBoxAndLandmarks, facechip);

    std::vector<TFFacechip> facechips;
    for (size_t i = 0; i < batchSize; ++i) {
        facechips.push_back(facechip);
    }

    std::vector<MaskLabel> maskLabels;

    // Time the mask detector
    auto t1 = Clock::now();
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.detectMasks(facechips, maskLabels);
    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Average time mask detection: " << totalTime / numIterations / static_cast<float>(batchSize)
              << " ms | batch size = " << batchSize << " | " << numIterations << " iterations" << std::endl;

}

void benchmarkBlinkDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations) {
    // Initialize the SDK
    ConfigurationOptions options;
    options.modelsPath = "./";
    auto modelsPath = std::getenv("MODELS_PATH");
    if (modelsPath) {
        options.modelsPath = modelsPath;
    }
    options.gpuOptions = gpuOptions;
    options.smallestFaceHeight = 40;

    // Since we initialize the module, we do not need to discard the first inference time.
    InitializeModule initializeModule;
    initializeModule.faceDetector = true;
    initializeModule.liveness = true;
    options.initializeModule = initializeModule;

    SDK tfSdk(options);
    bool valid = tfSdk.setLicense(license);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        exit (EXIT_FAILURE);
    }

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

    // Time the blink detector
    auto t1 = Clock::now();
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.detectBlink(img, faceBoxAndLandmarks, blinkstate);
    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Average time blink detection: " << totalTime / numIterations
              << " ms | " << numIterations << " iterations" << std::endl;

}

void benchmarkSpoofDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations) {
    // Initialize the SDK
    ConfigurationOptions options;
    options.modelsPath = "./";
    auto modelsPath = std::getenv("MODELS_PATH");
    if (modelsPath) {
        options.modelsPath = modelsPath;
    }
    options.gpuOptions = gpuOptions;
    options.smallestFaceHeight = 40;

    // Since we initialize the module, we do not need to discard the first inference time.
    InitializeModule initializeModule;
    initializeModule.faceDetector = true;
    initializeModule.passiveSpoof = true;
    options.initializeModule = initializeModule;

    SDK tfSdk(options);
    bool valid = tfSdk.setLicense(license);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        exit (EXIT_FAILURE);
    }

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

    float spoofScore;
    SpoofLabel label;

    errorCode = tfSdk.detectSpoof(img, faceBoxAndLandmarks, label, spoofScore);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Spoof function failed" << std::endl;
        std::cout << errorCode << std::endl;
        return;
    }

    // Time the spoof detector
    auto t1 = Clock::now();
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.detectSpoof(img, faceBoxAndLandmarks, label, spoofScore);
    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Average time spoof detection: " << totalTime / numIterations
              << " ms | " << numIterations << " iterations" << std::endl;

}

void benchmarkFaceLandmarkDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations) {
    // Initialize the SDK
    ConfigurationOptions options;
    options.modelsPath = "./";
    auto modelsPath = std::getenv("MODELS_PATH");
    if (modelsPath) {
        options.modelsPath = modelsPath;
    }
    options.gpuOptions = gpuOptions;
    options.smallestFaceHeight = 40;

    // Since we initialize the module, we do not need to discard the first inference time.
    InitializeModule initializeModule;
    initializeModule.faceDetector = true;
    options.initializeModule = initializeModule;

    SDK tfSdk(options);
    bool valid = tfSdk.setLicense(license);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        exit (EXIT_FAILURE);
    }

    // Load the image
    TFImage img;
    ErrorCode errorCode = tfSdk.preprocessImage("../images/headshot.jpg", img);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: could not load the image" << std::endl;
        return;
    }

    FaceBoxAndLandmarks faceBoxAndLandmarks;
    bool found = false;

    // Time the face detection
    auto t1 = Clock::now();
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.detectLargestFace(img, faceBoxAndLandmarks, found);
    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Average time face and landmark detection: " << totalTime / numIterations
              << " ms | " << numIterations << " iterations" << std::endl;

}
