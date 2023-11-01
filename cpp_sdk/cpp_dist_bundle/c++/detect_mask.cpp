// Sample code: load an image, detect the largest face and check whether the face has a mask on or
// not First image is of a person wearing a mask. Second image is of a person not wearing a mask.
// The probability that a mask is worn over the face is computed in both cases.

#include "tf_sdk.h"
#include <iostream>

using namespace Trueface;

int main() {
    // Start by specifying the configuration options to be used.
    // Can choose to use default configuration options if preferred by calling the default SDK
    // constructor. Learn more about configuration options here:
    // https://reference.trueface.ai/cpp/dev/latest/usage/general.html
    ConfigurationOptions options;
    // The face recognition model to use. TFV5_2 balances accuracy and speed.
    options.frModel = FacialRecognitionModel::TFV5_2;
    // The object detection model to use.
    options.objModel = ObjectDetectionModel::ACCURATE;
    // The face detection filter.
    options.fdFilter = FaceDetectionFilter::BALANCED;
    // Smallest face height in pixels for the face detector.
    options.smallestFaceHeight = 40;
    // The path specifying the directory where the model files have been downloaded
    options.modelsPath = "./";
    auto modelsPath = std::getenv("MODELS_PATH");
    if (modelsPath) {
        options.modelsPath = modelsPath;
    }
    // Enable vector compression to improve 1 to 1 comparison speed and 1 to N search speed.
    options.frVectorCompression = false;
    // Database management system for storage of biometric templates for 1 to N identification.
    options.dbms = DatabaseManagementSystem::SQLITE;

    // If encryption is enabled, must provide an encryption key
    options.encryptDatabase.enableEncryption = false;
    options.encryptDatabase.key = "TODO: Your encryption key here";

    // Initialize module in SDK constructor.
    // By default, the SDK uses lazy initialization, meaning modules are only initialized when they
    // are first used (on first inference). This is done so that modules which are not used do not
    // load their models into memory, and hence do not utilize memory. The downside to this is that
    // the first inference will be much slower as the model file is being decrypted and loaded into
    // memory. Therefore, if you know you will use a module, choose to pre-initialize the module,
    // which reads the model file into memory in the SDK constructor.
    InitializeModule initializeModule;
    initializeModule.faceDetector = true;
    initializeModule.faceRecognizer = true;
    options.initializeModule = initializeModule;

    // Options for enabling GPU
    // We will disable GPU inference, but you can easily enable it by modifying the following
    // options Note, you may require a specific GPU enabled token in order to enable GPU inference.
    options.gpuOptions = false; // TODO: Change this to true to enable GPU inference
    options.gpuOptions.deviceIndex = 0;

    GPUModuleOptions moduleOptions;
    moduleOptions.maxBatchSize = 4;
    moduleOptions.optBatchSize = 1;
    moduleOptions.maxWorkspaceSizeMb = 2000;
    moduleOptions.precision = Precision::FP16;

    options.gpuOptions.faceRecognizerGPUOptions = moduleOptions;
    options.gpuOptions.faceDetectorGPUOptions = moduleOptions;
    options.gpuOptions.maskDetectorGPUOptions = moduleOptions;
    options.gpuOptions.objectDetectorGPUOptions = moduleOptions;

    SDK tfSdk(options);
    // TODO: Either input your token in the CMakeLists.txt file, or insert it below directly
    bool valid = tfSdk.setLicense(TRUEFACE_TOKEN);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        return 1;
    }

    // Load the mask image and detect largest face.
    std::cout << "Image with mask" << std::endl;
    TFImage img;
    ErrorCode errorCode = tfSdk.preprocessImage("../../images/mask.jpg", img);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

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

    // Run mask detection
    MaskLabel maskLabel;
    float maskScore;
    errorCode = tfSdk.detectMask(img, faceBoxAndLandmarks, maskLabel, maskScore);

    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: could not run mask detection" << std::endl;

        return 1;
    }

    if (maskLabel == MaskLabel::MASK) {
        std::cout << "Mask detected with probability of " << 1.0f - maskScore << std::endl;
    } else {
        std::cout << "No mask detected with probability of " << maskScore << std::endl;
    }

    // Load the non mask image and detect largest face.
    std::cout << "\nImage without mask:" << std::endl;
    errorCode = tfSdk.preprocessImage("../../images/headshot.jpg", img);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

    errorCode = tfSdk.detectLargestFace(img, faceBoxAndLandmarks, found);

    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

    if (!found) {
        std::cout << "Unable to detect face in image 2" << std::endl;
        return 1;
    }

    // Run mask detection
    errorCode = tfSdk.detectMask(img, faceBoxAndLandmarks, maskLabel, maskScore);

    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: could not run mask detection" << std::endl;
        std::cout << errorCode << std::endl;
        return 1;
    }

    if (maskLabel == MaskLabel::MASK) {
        std::cout << "Mask detected with probability of " << 1.0f - maskScore << std::endl;
    } else {
        std::cout << "No mask detected with probability of " << maskScore << std::endl;
    }
}