// The following code runs speed benchmarks for the different modules
// The first few inferences are discarded to ensure caching is hot

#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>

#include "tf_sdk.h"
#include <fstream>

using namespace Trueface;

typedef std::chrono::high_resolution_clock Clock;

// Stopwatch Utility
template <typename Clock = std::chrono::high_resolution_clock>
class Stopwatch
{
    typename Clock::time_point start_point;
public:
    Stopwatch() :start_point(Clock::now()){}

    // Returns elapsed time
    template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration>
    Rep elapsedTime() const {
        std::atomic_thread_fence(std::memory_order_relaxed);
        auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start_point).count();
        std::atomic_thread_fence(std::memory_order_relaxed);
        return static_cast<Rep>(counted_time);
    }
};

using preciseStopwatch = Stopwatch<>;
using systemStopwatch = Stopwatch<std::chrono::system_clock>;
using monotonicStopwatch = Stopwatch<std::chrono::steady_clock>;

void benchmarkFaceRecognition(const std::string& license, FacialRecognitionModel model, const GPUOptions& gpuOptions, unsigned int batchSize = 1, unsigned int numIterations = 100);
void benchmarkObjectDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations = 100);
void benchmarkFaceLandmarkDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations = 100);
void benchmarkDetailedLandmarkDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations = 100);
void benchmarkPreprocessImage(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations = 100);
void benchmarkMaskDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int batchSize = 1, unsigned int numIterations = 100);
void benchmarkBlinkDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations = 100);
void benchmarkSpoofDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations = 100);
void benchmarkHeadOrientation(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations = 500);
void benchmarkFaceImageBlurDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations = 200);

bool warmup = true; // Warmup inference to ensure caching is hot
int numWarmup = 10;

int main() {
    const std::string license = TRUEFACE_TOKEN;

    GPUOptions gpuOptions;
    gpuOptions.enableGPU = false; // TODO set this to true to benchmark on GPU
    gpuOptions.deviceIndex = 0;

    GPUModuleOptions gpuModuleOptions;
    gpuModuleOptions.precision = Precision::FP16;

    int32_t batchSize = 4;
    gpuModuleOptions.maxBatchSize = batchSize;
    gpuModuleOptions.optBatchSize = batchSize;

    gpuOptions.faceDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceRecognizerGPUOptions = gpuModuleOptions;
    gpuOptions.maskDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.objectDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceLandmarkDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceOrientationDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceBlurDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.spoofDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.blinkDetectorGPUOptions = gpuModuleOptions;

    if (gpuOptions.enableGPU) {
        std::cout << "Using GPU for inference" << std::endl;
    } else {
        std::cout << "Using CPU for inference" << std::endl;
    }

    int multFactor = 1;
    if (gpuOptions.enableGPU) {
        multFactor = 10;
    }

    benchmarkPreprocessImage(license, gpuOptions, 200);
    benchmarkFaceLandmarkDetection(license, gpuOptions, 100 * multFactor);
    benchmarkDetailedLandmarkDetection(license, gpuOptions, 100 * multFactor);
    benchmarkHeadOrientation(license, gpuOptions, 500 * multFactor);
    benchmarkFaceImageBlurDetection(license, gpuOptions, 200 * multFactor);
    benchmarkBlinkDetection(license, gpuOptions, 100 * multFactor);
    benchmarkMaskDetection(license, gpuOptions, 1, 100 * multFactor);
    benchmarkSpoofDetection(license, gpuOptions, 100 * multFactor);
    benchmarkObjectDetection(license, gpuOptions, 100 * multFactor);

    if (!gpuOptions.enableGPU) {
        // Trueface::SDK::getFaceFeatureVectors is not supported by the LITE and LITE_V2 models.
        benchmarkFaceRecognition(license, FacialRecognitionModel::LITE, gpuOptions, 1,  200);
        benchmarkFaceRecognition(license, FacialRecognitionModel::LITE_V2, gpuOptions, 1,  200);
    }
    benchmarkFaceRecognition(license, FacialRecognitionModel::TFV7, gpuOptions, 1,  40 * multFactor);
    benchmarkFaceRecognition(license, FacialRecognitionModel::TFV6, gpuOptions, 1,  40 * multFactor);
    benchmarkFaceRecognition(license, FacialRecognitionModel::TFV5_2, gpuOptions, 1,  40 * multFactor);

    // Benchmarks with batching.
    // On CPU, should be the same speed as a batch size of 1.
    // On GPU, will increase the throughput.
    benchmarkFaceRecognition(license, FacialRecognitionModel::TFV5_2, gpuOptions, batchSize, 40 * multFactor);
    benchmarkFaceRecognition(license, FacialRecognitionModel::TFV6, gpuOptions, batchSize, 40 * multFactor);
    benchmarkFaceRecognition(license, FacialRecognitionModel::TFV7, gpuOptions, batchSize, 40 * multFactor);
    benchmarkMaskDetection(license, gpuOptions, batchSize, 100 * multFactor);

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
    for (size_t i = 0; i < batchSize; ++i) {
        facechips.push_back(facechip);
    }

    std::vector<Faceprint> faceprints;

    if (warmup) {
        for (int i = 0; i < numWarmup; ++i) {
            errorCode = tfSdk.getFaceFeatureVectors(facechips, faceprints);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to run face recognition" << std::endl;
                return;
            }
        }
    }

    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.getFaceFeatureVectors(facechips, faceprints);
    }

    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

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
    options.gpuOptions = gpuOptions;
    options.objModel = ObjectDetectionModel::FAST;

    // Since we initialize the module, we do not need to discard the first inference time.
    InitializeModule initializeModule;
    initializeModule.objectDetector = true;
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

    if (warmup) {
        for (int i = 0; i < numWarmup; ++i) {
            errorCode = tfSdk.detectObjects(img, boundingBoxes);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Could not run object detection!" << std::endl;
                return;
            }
        }
    }

    // Time the creation of the feature vector
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.detectObjects(img, boundingBoxes);

    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

    const std::string mode = (options.objModel == ObjectDetectionModel::FAST) ? "fast" : "accurate";

    std::cout << "Average time object detection (" + mode + " mode): " << totalTime / numIterations
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

    const std::string imgPath = "../images/headshot.jpg";

    // First run the benchmark for an image on disk
    // Run once to ensure everything works
    TFImage img;
    if (warmup) {
        for (int i = 0; i < numWarmup; ++i) {
            auto errorcode = tfSdk.preprocessImage(imgPath, img);
            if (errorcode != ErrorCode::NO_ERROR) {
                std::cout << "Unable to preprocess the image" << std::endl;
                return;
            }
        }
    }

    // Time the preprocessImage function
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIterations; ++i) {
        TFImage newImg;
        tfSdk.preprocessImage(imgPath, img);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

    std::cout << "Average time preprocessImage JPG image from disk (" << img->getWidth() << "x" << img->getHeight() << "): " <<
              totalTime / numIterations << " ms | " << numIterations << " iterations" << std::endl;

    // Now repeat with encoded image in memory
    std::ifstream file(imgPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cout << "Unable to load the image" << std::endl;
        return;
    }

    if (warmup) {
        for (int i = 0; i < numWarmup; ++i) {
            auto errorcode = tfSdk.preprocessImage(buffer, img);
            if (errorcode != ErrorCode::NO_ERROR) {
                std::cout << "Unable to preprocess the image" << std::endl;
                return;
            }
        }
    }

    // Time the preprocessImage function
    preciseStopwatch stopwatch1;
    for (size_t i = 0; i < numIterations; ++i) {
        TFImage newImg;
        tfSdk.preprocessImage(buffer, newImg);
    }
    totalTime = stopwatch1.elapsedTime<float, std::chrono::milliseconds>();

    std::cout << "Average time preprocessImage encoded JPG image in memory (" << img->getWidth() << "x" << img->getHeight() << "): " <<
    totalTime / numIterations << " ms | " << numIterations << " iterations" << std::endl;

    // Now repeat the same experiment with a decoded image
    buffer.resize(img->getHeight() * img->getWidth() * img->getChannels());
    const auto height = img->getHeight();
    const auto width = img->getWidth();
    memcpy(buffer.data(), img->getData(), buffer.size());

    if (warmup) {
        for (int i = 0; i < 10; ++i) {
            auto errorCode = tfSdk.preprocessImage(buffer.data(), width, height, ColorCode::rgb, img);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to preprocess image" << std::endl;
                return;
            }
        }
    }

    preciseStopwatch stopwatch2;
    for (size_t i = 0; i < numIterations; ++i) {
        TFImage newImg;
        tfSdk.preprocessImage(buffer, newImg);
    }
    totalTime = stopwatch2.elapsedTime<float, std::chrono::microseconds>();

    std::cout << "Average time preprocessImage RGB pixel array in memory (" << img->getWidth() << "x" << img->getHeight() << "): " <<
              totalTime / numIterations << " us | " << numIterations << " iterations" << std::endl;
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

    if (warmup) {
        for (int i = 0; i < numWarmup; ++i) {
            tfSdk.getFaceLandmarks(img, faceBoxAndLandmarks, landmarks);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to get detailed landmarks" << std::endl;
                return;
            }
        }
    }

    // Time the landmark detection
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.getFaceLandmarks(img, faceBoxAndLandmarks, landmarks);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

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

    Landmarks landmarks;
    errorCode = tfSdk.getFaceLandmarks(img, faceBoxAndLandmarks, landmarks);

    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to detect landmarks" << std::endl;
        return;
    }

    float yaw, pitch, roll;
    std::array<double, 3> rotationVec, translationVec;

    if (warmup) {
        for (int i = 0; i < numWarmup; ++i) {
            errorCode = tfSdk.estimateHeadOrientation(img, faceBoxAndLandmarks, landmarks, yaw, pitch, roll, rotationVec, translationVec);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Unable to run head orientation method" << std::endl;
                return;
            }
        }
    }

    // Time the head orientation
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.estimateHeadOrientation(img, faceBoxAndLandmarks, landmarks, yaw, pitch, roll, rotationVec, translationVec);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

    std::cout << "Average time head orientation: " << totalTime / numIterations
              << " ms | " << numIterations << " iterations" << std::endl;


}

void benchmarkFaceImageBlurDetection(const std::string& license, const GPUOptions& gpuOptions, unsigned int numIterations ) {
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
    initializeModule.faceBlurDetector = true;
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
    errorCode = tfSdk.extractAlignedFace(img, faceBoxAndLandmarks, facechip);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to extract aligned face for mask detection" << std::endl;
        return;
    }

    FaceImageQuality quality;

    if (warmup) {
        for (int i = 0; i < numWarmup; ++i) {
            errorCode = tfSdk.detectFaceImageBlur(facechip, quality);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to detect face image blur" << std::endl;
                return;
            }
        }
    }

    // Time the mask detector
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.detectFaceImageBlur(facechip, quality);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

    std::cout << "Average time face image blur detection: " << totalTime / numIterations
              << " ms  | " << numIterations << " iterations" << std::endl;

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
    errorCode = tfSdk.extractAlignedFace(img, faceBoxAndLandmarks, facechip);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to extract aligned face for mask detection" << std::endl;
        return;
    }

    std::vector<TFFacechip> facechips;
    for (size_t i = 0; i < batchSize; ++i) {
        facechips.push_back(facechip);
    }

    std::vector<MaskLabel> maskLabels;

    if (warmup) {
        for (int i = 0; i < numWarmup; ++i) {
            errorCode = tfSdk.detectMasks(facechips, maskLabels);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to run mask detection" << std::endl;
                return;
            }
        }
    }

    // Time the mask detector
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.detectMasks(facechips, maskLabels);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

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
    initializeModule.blinkDetector = true;
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

    if (warmup) {
        for (int i = 0; i < numWarmup; ++i) {
            errorCode = tfSdk.detectBlink(img, faceBoxAndLandmarks, blinkstate);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to run blink detection" << std::endl;
                return;
            }
        }
    }

    // Time the blink detector
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.detectBlink(img, faceBoxAndLandmarks, blinkstate);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

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

    if (warmup) {
        for (int i = 0; i < numWarmup; ++i) {
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
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.detectSpoof(img, faceBoxAndLandmarks, label, spoofScore);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

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

    if (warmup) {
        for (int i = 0; i < numWarmup; ++i) {
            errorCode = tfSdk.detectLargestFace(img, faceBoxAndLandmarks, found);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to run face detection" << std::endl;
                return;
            }
        }
    }

    // Time the face detection
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIterations; ++i) {
        tfSdk.detectLargestFace(img, faceBoxAndLandmarks, found);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();

    std::cout << "Average time face and landmark detection: " << totalTime / numIterations
              << " ms | " << numIterations << " iterations" << std::endl;

}
