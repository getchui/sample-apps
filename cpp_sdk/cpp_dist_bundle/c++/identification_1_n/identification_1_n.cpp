// Sample code: Load collection from disk then run 1N identification

// This sample app demonstrates how to use 1N identification. First, an existing collection (created by running enroll_in_database) is loaded from disk.
// Next, 1N identification is run to determine the identity of an anonymous template.
// Note, you must run enroll_in_database before being able to run this sample app. 

#include "tf_sdk.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>

using namespace Trueface;

int main() {
    // Start by specifying the configuration options to be used.
    // Can choose to use default configuration options if preferred by calling the default SDK constructor.
    // Learn more about configuration options here: https://reference.trueface.ai/cpp/dev/latest/usage/general.html
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
    options.gpuOptions.objectDetectorGPUOptions = moduleOptions;

    SDK tfSdk(options);
    // TODO: Either input your token in the CMakeLists.txt file, or insert it below directly
    bool valid = tfSdk.setLicense(TRUEFACE_TOKEN);

    if (valid == false) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        return 1;
    }

    // The name of our database, and the name of our collection in that database.
    const std::string databaseName = "my_database.db";
    const std::string collectionName = "my_collection";

    // Connect to the existing database
    auto retcode = tfSdk.createDatabaseConnection(databaseName);

    // If using the POSTGRESQL backend, then it might look something like this...
//    auto retcode = tfSdk.createDatabaseConnection("host=localhost port=5432 dbname=my_database user=postgres password=admin");

    if (retcode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to create database connection\n";
        std::cout << retcode << std::endl;
        return -1;
    }

    // Load the existing collection into memory.
    retcode = tfSdk.createLoadCollection(collectionName);
    if (retcode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to create or load collection\n";
        std::cout << retcode << std::endl;
        return -1;
    }

    // Use image of Brad Pitt as probe image
    TFImage img;
    retcode = tfSdk.preprocessImage("../../images/brad_pitt_1.jpg", img);
    if (retcode != ErrorCode::NO_ERROR) {
        std::cout << retcode << std::endl;
        return -1;
    }

    // Generate a template as the probe
    // We do not need to ensure that the probe template is high quality.
    // Only the enrollment templates must be high quality.
    Faceprint probeFaceprint;
    bool foundFace;
    retcode = tfSdk.getLargestFaceFeatureVector(img, probeFaceprint, foundFace);
    if (retcode != ErrorCode::NO_ERROR) {
        std::cout << retcode << std::endl;
        return -1;
    }

    if (!foundFace) {
        std::cout << "Unable to find face in image\n";
        return -1;
    }

    // Run identify function
    Candidate candidate;
    bool found;

    // TODO: Select a threshold for your application using the ROC curves
    // https://docs.trueface.ai/roc-curves
    const float threshold = 0.4;
    retcode = tfSdk.identifyTopCandidate(probeFaceprint, candidate, found, threshold);

    if (retcode != ErrorCode::NO_ERROR) {
        std::cout << "There was an error with the call to identify" << std::endl;
        std::cout << retcode << std::endl;
        return -1;
    }

    if (!found) {
        std::cout << "Unable to find match" << std::endl;
        return -1;
    }

    std::cout << "Identity found: " << candidate.identity << std::endl;
    std::cout << "Match Similarity: " << candidate.similarityMeasure << std::endl;
    std::cout << "Match Probability: " << candidate.matchProbability * 100 << "%" << std::endl;

    return 0;
}


