#include <iostream>
#include <vector>

#include "tf_sdk.h"
#include "tf_data_types.h"

using namespace Trueface;

int main() {
    // Start by specifying the configuration options to be used.
    // Can choose to use default configuration options if preferred by calling the default SDK constructor.
    // Learn more about configuration options here: https://reference.trueface.ai/cpp/dev/latest/usage/general.html
    ConfigurationOptions options;
    // The face recognition model to use. Balances accuracy and speed.
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

    SDK tfSdk (options);

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense(TRUEFACE_TOKEN);
    if (!isValid) {
        std::cout << "Error: the provided license is invalid\n";
        return -1;
    }

    // Load the first image of Obama
    TFImage img;
    auto errorCode = tfSdk.preprocessImage("../../../../images/obama/obama1.jpg", img);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: unable to read image\n";
        return -1;
    }

    // Generate a template from the first image
    Faceprint faceprint1;
    bool found;
    errorCode = tfSdk.getLargestFaceFeatureVector(img, faceprint1, found);
    if (errorCode != ErrorCode::NO_ERROR || !found) {
        std::cout << "Error: Unable to generate template\n";
        return -1;
    }

    // Load the second image of obama
    errorCode = tfSdk.preprocessImage("../../../../images/obama/obama2.jpg", img);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: uanble to read image\n";
        return -1;
    }

    // Generate a template from the second image
    Faceprint faceprint2;
    errorCode = tfSdk.getLargestFaceFeatureVector(img, faceprint2, found);
    if (errorCode != ErrorCode::NO_ERROR || !found) {
        std::cout << "Error: Unable to generate template\n";
        return -1;
    }

    // Compare two images of Obama
    float matchProbabilitiy, similarityMeasure;
    errorCode = tfSdk.getSimilarity(faceprint1, faceprint2, matchProbabilitiy, similarityMeasure);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to generate similarity score\n";
        return -1;
    }

    std::cout << "Similarity between two Obama images: " << matchProbabilitiy << "\n";

    // Load image of armstrong
    errorCode = tfSdk.preprocessImage("../../../../images/armstrong/armstrong1.jpg", img);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: uanble to read image\n";
        return -1;
    }

    // Generate a template from the second third
    Faceprint faceprint3;
    errorCode = tfSdk.getLargestFaceFeatureVector(img, faceprint3, found);
    if (errorCode != ErrorCode::NO_ERROR || !found) {
        std::cout << "Error: Unable to generate template\n";
        return -1;
    }

    // Compare the image of Obama to Armstrong
    errorCode = tfSdk.getSimilarity(faceprint1, faceprint3, matchProbabilitiy, similarityMeasure);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to generate similarity score\n";
        return -1;
    }

    std::cout << "Similarity between Obama and Armstrong: " << matchProbabilitiy << "\n";

    return 0;
}