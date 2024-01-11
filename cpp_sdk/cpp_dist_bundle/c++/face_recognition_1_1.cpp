// Sample code: Initialize the SDK using TFV5 model, load two images, extract feature vectors and
// compare the similarity First two images of the same identity are compared. Then the similarity
// score of two different identities is computed.

#include "tf_sdk.h"
#include <iostream>
#include <vector>

using namespace Trueface;

int main() {
    // Start by specifying the configuration options to be used.
    // Can choose to use default configuration options if preferred by calling the default SDK
    // constructor. Learn more about configuration options here:
    // https://reference.trueface.ai/cpp/dev/latest/usage/general.html
    ConfigurationOptions options;
    // For the sake of this sample app, will demonstrate how to set all the configuration options.
    // Some of them may not be relevant to this sample app.
    // The face recognition model to use. TFV5_2 balances accuracy and speed.
    options.frModel = FacialRecognitionModel::TFV5_2;
    // The face detection model
    options.fdModel = FaceDetectionModel::FAST;
    // The face detection filter.
    options.fdFilter = FaceDetectionFilter::BALANCED;
    // Smallest face height in pixels for the face detector.
    options.smallestFaceHeight = 40;
    // Use a global inference threadpool
    options.useGlobalInferenceThreadpool = true;
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
    initializeModule.faceRecognizer = true;
    initializeModule.faceDetector = true;
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
    options.gpuOptions.faceLandmarkDetectorGPUOptions = moduleOptions;
    options.gpuOptions.faceDetectorGPUOptions = moduleOptions;
    options.gpuOptions.maskDetectorGPUOptions = moduleOptions;
    options.gpuOptions.objectDetectorGPUOptions = moduleOptions;
    options.gpuOptions.faceOrientationDetectorGPUOptions = moduleOptions;
    options.gpuOptions.faceBlurDetectorGPUOptions = moduleOptions;
    options.gpuOptions.spoofDetectorGPUOptions = moduleOptions;
    options.gpuOptions.blinkDetectorGPUOptions = moduleOptions;
    options.gpuOptions.faceTemplateQualityEstimatorGPUOptions = moduleOptions;
    SDK tfSdk(options);
    // TODO: Either input your token in the CMakeLists.txt file, or insert it below directly
    bool valid = tfSdk.setLicense(TRUEFACE_TOKEN);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        return 1;
    }

    // Load all our images at once
    TFImage img1, img2, img3;
    auto errorCode = tfSdk.preprocessImage("../../images/brad_pitt_1.jpg", img1);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

    errorCode = tfSdk.preprocessImage("../../images/brad_pitt_2.jpg", img2);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

    errorCode = tfSdk.preprocessImage("../../images/tom_cruise_1.jpg", img3);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

    // Generate the face recognition feature vectors
    Faceprint faceprint1, faceprint2, faceprint3;
    bool foundFace;
    errorCode = tfSdk.getLargestFaceFeatureVector(img1, faceprint1, foundFace);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

    if (!foundFace) {
        std::cout << "Unable to find face in image" << std::endl;
        return 1;
    }

    errorCode = tfSdk.getLargestFaceFeatureVector(img2, faceprint2, foundFace);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

    if (!foundFace) {
        std::cout << "Unable to find face in image" << std::endl;
        return 1;
    }

    errorCode = tfSdk.getLargestFaceFeatureVector(img3, faceprint3, foundFace);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

    if (!foundFace) {
        std::cout << "Unable to find face in image" << std::endl;
        return 1;
    }

    // Compute the similarity between the images of the same identity.
    float similarityScore;
    float matchProbability;
    errorCode = SDK::getSimilarity(faceprint1, faceprint2, matchProbability, similarityScore);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute similarity score" << std::endl;
        return 1;
    }

    std::cout << "Similarity score of same identity images: " << similarityScore << std::endl;
    std::cout << "Match probability of same identity images: " << matchProbability * 100 << "% \n"
              << std::endl;

    // Compute the similarity between the images of different identities.
    errorCode = SDK::getSimilarity(faceprint1, faceprint3, matchProbability, similarityScore);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute similarity score" << std::endl;
        return 1;
    }

    std::cout << "Similarity score of two different identities: " << similarityScore << std::endl;
    std::cout << "Match probability of two different identities: " << matchProbability * 100 << "%"
              << std::endl;

    return 0;
}