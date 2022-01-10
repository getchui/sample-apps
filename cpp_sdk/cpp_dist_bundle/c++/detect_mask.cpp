// Sample code: load an image, detect the largest face and check whether the face has a mask on or not
// First image is of a person wearing a mask. Second image is of a person not wearing a mask.
// The probability that a mask is worn over the face is computed in both cases.

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
    initializeModule.faceDetector = true;
    initializeModule.faceRecognizer = true;
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
        std::cout<<"Error: the provided license is invalid."<<std::endl;
        return 1;
    }

    // Load the mask image and detect largest face.
    std::cout << "Image with mask" << std::endl;
    ErrorCode errorCode = tfSdk.setImage("../../images/mask.jpg");
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout<<"Error: could not load the image"<<std::endl;
        return 1;
    }

    FaceBoxAndLandmarks faceBoxAndLandmarks;
    bool found = false;
    errorCode = tfSdk.detectLargestFace(faceBoxAndLandmarks, found);

    if (!found || errorCode != ErrorCode::NO_ERROR) {
        std::cout<<"Error: face not found";
        return 1;
    }

    // Run mask detection
    MaskLabel maskLabel;
    errorCode = tfSdk.detectMask(faceBoxAndLandmarks, maskLabel);

    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout<<"Error: could not run mask detection"<<std::endl;
        return 1;
    }

    if (maskLabel == MaskLabel::MASK) {
        std::cout << "Mask detected" << std::endl;
    } else {
        std::cout << "No mask detected" << std::endl;
    }

    // Load the non mask image and detect largest face.
    std::cout << "\nImage without mask:" << std::endl;
    errorCode = tfSdk.setImage("../../images/headshot.jpg");
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout<<"Error: could not load the image"<<std::endl;
        return 1;
    }

    errorCode = tfSdk.detectLargestFace(faceBoxAndLandmarks, found);

    if (!found || errorCode != ErrorCode::NO_ERROR) {
        std::cout<<"Error: face not found";
        return 1;
    }

    // Run mask detection
    errorCode = tfSdk.detectMask(faceBoxAndLandmarks, maskLabel);

    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout<<"Error: could not run mask detection"<<std::endl;
        return 1;
    }

    if (maskLabel == MaskLabel::MASK) {
        std::cout << "Mask detected" << std::endl;
    } else {
        std::cout << "No mask detected" << std::endl;
    }
}