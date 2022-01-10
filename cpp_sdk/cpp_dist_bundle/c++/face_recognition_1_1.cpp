// Sample code: Initialize the SDK using TFV5 model, load two images, extract feature vectors and compare the similarity
// First two images of the same identity are compared. Then the similarity score of two different identities is computed.

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
    // Enable vector compression to improve 1 to 1 comparison speed and 1 to N search speed.
    options.frVectorCompression = false;
    // Database management system for storage of biometric templates for 1 to N identification.
    options.dbms = DatabaseManagementSystem::SQLITE;

    // Choose to encrypt the database
    EncryptDatabase encryptDatabase;
    encryptDatabase.enableEncryption = false; // TODO: To encrypt the database change this to true
    encryptDatabase.key = "TODO: Your encryption key here";
    options.encryptDatabase = encryptDatabase;

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

    // Load the first image and extract the feature vector.
    auto errorCode = tfSdk.setImage("../../images/brad_pitt_1.jpg");
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: could not load the image" << std::endl;
        return 1;
    }

    Faceprint faceprint1;
    bool foundFace;
    errorCode = tfSdk.getLargestFaceFeatureVector(faceprint1, foundFace);
    if (errorCode != ErrorCode::NO_ERROR || !foundFace) {
        std::cout << "Error: Unable to detect face in image" << std::endl;
        return 1;
    }

    // Load the second image and extract the feature vector.
    errorCode = tfSdk.setImage("../../images/brad_pitt_2.jpg");
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: could not load the image 2" << std::endl;
        return 1;
    }

    Faceprint faceprint2;
    errorCode = tfSdk.getLargestFaceFeatureVector(faceprint2, foundFace);
    if (errorCode != ErrorCode::NO_ERROR || !foundFace) {
        std::cout << "Error: Unable to detect face in image" << std::endl;
        return 1;
    }

    // Compute the similarity between the two face images.
    float similarityScore;
    float matchProbability;
    errorCode = SDK::getSimilarity(faceprint1, faceprint2, matchProbability, similarityScore);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute similarity score" << std::endl;
        return 1;
    }

    std::cout<< "Similarity score of same identity images: " << similarityScore << std::endl;
    std::cout<< "Match probability of same identity images: " << matchProbability  << "\n" << std::endl;

    // Load the image of a different identity and extract the feature vector.
    errorCode = tfSdk.setImage("../../images/tom_cruise_1.jpg");
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: could not load the image 2" << std::endl;
        return 1;
    }

    Faceprint faceprint3;
    errorCode = tfSdk.getLargestFaceFeatureVector(faceprint3, foundFace);
    if (errorCode != ErrorCode::NO_ERROR || !foundFace) {
        std::cout << "Error: Unable to detect face in image" << std::endl;
        return 1;
    }

    // Compute the similarity between the images of different identities.
    errorCode = SDK::getSimilarity(faceprint1, faceprint3, matchProbability, similarityScore);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute similarity score" << std::endl;
        return 1;
    }

    std::cout << "Similarity score of two different identities: " << similarityScore << std::endl;
    std::cout << "Match probability of two different identities: " << matchProbability << std::endl;

    return 0;
}