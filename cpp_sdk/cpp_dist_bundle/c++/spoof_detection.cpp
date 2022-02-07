// Sample code: sets an image and runs spoof detection on the image

// This sample demonstrates how to use the spoof detector to identify presentation attacks.
// First, the image is set. Next, the spood detection function is used to identify potential spoof.


#include "tf_sdk.h"
#include <iostream>

using namespace Trueface;

int main() {
    // Start by specifying the configuration options to be used.
    // Can choose to use default configuration options if preferred by calling the default SDK constructor.
    // Learn more about configuration options here: https://reference.trueface.ai/cpp/dev/latest/usage/general.html
    ConfigurationOptions options;
    // The face recognition model to use. Use the most accurate face recognition model.
    options.frModel = FacialRecognitionModel::TFV5;
    // The object detection model to use.
    options.objModel = ObjectDetectionModel::ACCURATE;
    // The face detection filter.
    options.fdFilter = FaceDetectionFilter::BALANCED;
    // Smallest face height in pixels for the face detector.
    options.smallestFaceHeight = 40;
    // The path specifying the directory where the model files have been downloaded
    options.modelsPath = "./";
    // Enable vector compression to improve 1 to 1 comparison speed and 1 to N search speed.
    options.frVectorCompression = false;
    // Database management system for storage of biometric templates for 1 to N identification.
    options.dbms = DatabaseManagementSystem::SQLITE;

    // If encryption is enabled, must provide an encryption key
    options.encryptDatabase.enableEncryption = false;
    options.encryptDatabase.key = "TODO: Your encryption key here";

    // Initialize module in SDK constructor.
    // By default, the SDK uses lazy initialization, meaning modules are only initialized when they are first used (on first inference).
    // This is done so that modules which are not used do not load their models into memory, and hence do not utilize memory.
    // The downside to this is that the first inference will be much slower as the model file is being decrypted and loaded into memory.
    // Therefore, if you know you will use a module, choose to pre-initialize the module, which reads the model file into memory in the SDK constructor.
    InitializeModule initializeModule;
    initializeModule.faceDetector = true;
    options.initializeModule = initializeModule;

    // Options for enabling GPU
    // We will disable GPU inference, but you can easily enable it by modifying the following options
    // Note, you may require a specific GPU enabled token in order to enable GPU inference.
    GPUModuleOptions gpuOptions;
    gpuOptions.enableGPU = false; // TODO: Change this to true to enable GPU inference.
    gpuOptions.maxBatchSize = 4;
    gpuOptions.optBatchSize = 1;
    gpuOptions.maxWorkspaceSizeMb = 2000;
    gpuOptions.deviceIndex = 0;
    gpuOptions.precision = Precision::FP16;

    options.gpuOptions.faceRecognizerGPUOptions = gpuOptions;
    options.gpuOptions.faceDetectorGPUOptions = gpuOptions;

    // Alternatively, can also do the following to enable GPU inference for all supported modules:
//    options.gpuOptions = true;
    
    SDK tfSdk(options);
    // TODO: Either input your token in the CMakeLists.txt file, or insert it below directly
    bool valid = tfSdk.setLicense(TRUEFACE_TOKEN);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        return 1;
    }

    // Load the real image
    TFImage img;
    ErrorCode errorCode = tfSdk.preprocessImage("../../images/real.jpg", img);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

    std::cout << "Running spoof detection with real image..." << std::endl;
    FaceBoxAndLandmarks faceBoxAndLandmarks;
    bool found = false;
    errorCode = tfSdk.detectLargestFace(img, faceBoxAndLandmarks, found);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

    if (!found) {
        std::cout << "Unable to find face in image 1" << std::endl;
        return 1;
    }

    SpoofLabel result;
    float spoofScore;

    errorCode = tfSdk.detectSpoof(img, faceBoxAndLandmarks, result, spoofScore);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: could not run spoof detection" << std::endl;
        std::cout << errorCode << std::endl;
        return 1;
    }

    if (result == SpoofLabel::REAL) {
        std::cout << "Real image detected" << std::endl;
    } else {
        std::cout << "Fake image detected" << std::endl;
    }

    // Load the fake image
    errorCode = tfSdk.preprocessImage("../../images/fake.jpg", img);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

    std::cout << "\nRunning spoof detection with fake image..." << std::endl;
    errorCode = tfSdk.detectLargestFace(img, faceBoxAndLandmarks, found);

    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

    if (!found) {
        std::cout << "Unable to find face in image 2" << std::endl;
    }
    errorCode = tfSdk.detectSpoof(img, faceBoxAndLandmarks, result, spoofScore);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: could not run spoof detection" << std::endl;
        std::cout << errorCode << std::endl;
        return 1;
    }

    if (result == SpoofLabel::REAL) {
        std::cout << "Real image detected" << std::endl;
    } else {
        std::cout << "Fake image detected" << std::endl;
    }

    return 0;
}