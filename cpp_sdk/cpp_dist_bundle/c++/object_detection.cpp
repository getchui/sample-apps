// Sample code: load an image, run object detection
// This sample app demonstrates how to run object detection on an image
// An image of a person on a bike is first loaded. Next object detection is run and the predicted labels are printed.

#include "tf_sdk.h"
#include <iostream>
#include <vector>

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
    // By default, the SDK uses lazy initialization, meaning modules are only initialized when they are first used (on first inference).
    // This is done so that modules which are not used do not load their models into memory, and hence do not utilize memory.
    // The downside to this is that the first inference will be much slower as the model file is being decrypted and loaded into memory.
    // Therefore, if you know you will use a module, choose to pre-initialize the module, which reads the model file into memory in the SDK constructor.
    InitializeModule initializeModule;
    initializeModule.objectDetector = true;
    options.initializeModule = initializeModule;

    // Options for enabling GPU
    // We will disable GPU inference, but you can easily enable it by modifying the following options
    // Note, you may require a specific GPU enabled token in order to enable GPU inference.
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

    // Load the image of the person on a bike
    TFImage img;
    ErrorCode errorCode = tfSdk.preprocessImage("../../images/person_on_bike.jpg", img);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

    std::vector<BoundingBox> boundingBoxes;

    // Run object detection
    errorCode = tfSdk.detectObjects(img, boundingBoxes);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

    for (auto bbox : boundingBoxes) {
        // Convert the label to a string
        std::string label = tfSdk.getObjectLabelString(bbox.label);
        // Print out image label and probability
        std::cout << "Detected " << label << " with probability: " << bbox.probability << std::endl;
    }

    // Now annotate the image
    errorCode = tfSdk.drawObjectLabels(img, boundingBoxes);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return -1;
    }

    // Save the annotated image
    const std::string outputPath = "./annotated.jpg";
    img->saveImage(outputPath);

    std::cout << "Annotated image saved to: " << outputPath << std::endl;

    return 0;
}