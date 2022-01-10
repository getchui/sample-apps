// Sample code: load images and determine if a person is blinking in the images

// This sample demonstrates how to use the detectBlink function. First, an image is loaded.
// Next, the blink scores are computed.


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

    // Initialize module in SDK constructor.
    // By default, the SDK uses lazy initialization, meaning modules are only initialized when they are first used (on first inference).
    // This is done so that modules which are not used do not load their models into memory, and hence do not utilize memory.
    // The downside to this is that the first inference will be much slower as the model file is being decrypted and loaded into memory.
    // Therefore, if you know you will use a module, choose to pre-initialize the module, which reads the model file into memory in the SDK constructor.
    InitializeModule initializeModule;
    initializeModule.liveness = true;
    options.initializeModule = initializeModule;

    // Options for enabling GPU
    // We will disable GPU inference, but you can easily enable it by modifying the following options
    // Note, you may require a specific GPU enabled token in order to enable GPU inference.
    GPUModuleOptions gpuOptions;
    gpuOptions.enableGPU = false; // TODO: Change this to true to enable GPU inference.
    gpuOptions.maxBatchSize = 4;
    gpuOptions.optBatchSize = 1;
    gpuOptions.maxWorkspaceSizeMb = 2000;

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

    // Load the image with the eyes open
    ErrorCode errorCode = tfSdk.setImage("../../images/open_eyes.jpg");
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: could not load the image" << std::endl;
        return 1;
    }

    std::cout << "Running blink detection with open eye image" << std::endl;

    // Start by detecting the largest face in the image
    bool found;
    FaceBoxAndLandmarks fb;

    errorCode = tfSdk.detectLargestFace(fb, found);
    if (!found || errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to detect face in image 1" << std::endl;
        return 0;
    }

    // Compute if the detected face has eyes open or closed

    BlinkState blinkState;
    errorCode = tfSdk.detectBlink(fb, blinkState);
    if (errorCode == ErrorCode::EXTREME_FACE_ANGLE) {
        std::cout << "The face angle is too extreme! Please ensure face image is forward facing!" << std::endl;
        return 0;
    } else if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute blink!" << std::endl;
        return 0;
    }

    // At this point, we can use the members of BlinkState along with our own threshold to determine if the eyes are open or closed
    // Alternatively, we can use the pre-set thresholds by consulting BlinkState.isLeftEyeClosed and BlinkState.isRightEyeClosed

    std::cout << "Left eye score: " << blinkState.leftEyeScore << std::endl;
    std::cout << "Right eye score: " << blinkState.rightEyeScore << std::endl;
    std::cout << "Left eye aspect ratio: " << blinkState.leftEyeAspectRatio << std::endl;
    std::cout << "Right eye aspect ratio: " << blinkState.rightEyeAspectRatio << std::endl;

    if (blinkState.isLeftEyeClosed && blinkState.isRightEyeClosed) {
        std::cout << "PREDICTED RESULTS: Both eyes are closed!" << std::endl;
    } else {
        std::cout << "PREDICTED RESULTS:  The eyes are not closed!" << std::endl;
    }

    std::cout << "\nRunning blink detection with closed eye image" << std::endl;

    // Load the image with the eyes closed
    errorCode = tfSdk.setImage("../../images/closed_eyes.jpg");
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: could not load the image" << std::endl;
        return 1;
    }

    errorCode = tfSdk.detectLargestFace(fb, found);
    if (!found || errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to detect face in image 2" << std::endl;
        return 0;
    }

    // Compute if the detected face has eyes open or closed

    errorCode = tfSdk.detectBlink(fb, blinkState);
    if (errorCode == ErrorCode::EXTREME_FACE_ANGLE) {
        std::cout << "The face angle is too extreme! Please ensure face image is forward facing!" << std::endl;
        return 0;
    } else if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute blink!" << std::endl;
        return 0;
    }

    // At this point, we can use the members of BlinkState along with our own threshold to determine if the eyes are open or closed
    // Alternatively, we can use the pre-set thresholds by consulting BlinkState.isLeftEyeClosed and BlinkState.isRightEyeClosed

    std::cout << "Left eye score: " << blinkState.leftEyeScore << std::endl;
    std::cout << "Right eye score: " << blinkState.rightEyeScore << std::endl;
    std::cout << "Left eye aspect ratio: " << blinkState.leftEyeAspectRatio << std::endl;
    std::cout << "Right eye aspect ratio: " << blinkState.rightEyeAspectRatio << std::endl;

    if (blinkState.isLeftEyeClosed && blinkState.isRightEyeClosed) {
        std::cout << "PREDICTED RESULTS: Both eyes are closed!" << std::endl;
    } else {
        std::cout << "PREDICTED RESULTS: The eyes are not closed!" << std::endl;
    }

    return 0;
}