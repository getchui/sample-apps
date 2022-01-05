// Sample code: Load collection from disk then run 1N identification

// This sample app demonstrates how to use 1N identification. First, an existing collection (created by running enroll_in_database) is loaded from disk.
// Next, 1N identification is run to determine the identity of an anonymous template.
// Note, you must run enroll_in_database before being able to run this sample app. 

#include "tf_sdk.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>

int main() {
    // For a full list of configuration options, visit: https://reference.trueface.ai/cpp/dev/latest/usage/general.html#_CPPv4N8Trueface20ConfigurationOptionsE
    Trueface::ConfigurationOptions options;
    options.frModel = Trueface::FacialRecognitionModel::TFV5;
    // Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file
    options.dbms = Trueface::DatabaseManagementSystem::SQLITE; // Load the collection from an SQLITE database
    // TODO: If you have a NVIDIA gpu, then enable the enableGPU flag (you will require a GPU specific token for this).
//    options.gpuOptions = true;

    // To enable database encryption...
//    Trueface::EncryptDatabase encryptDatabase;
//    encryptDatabase.enableEncryption = true;
//    // TODO: Replace with your own encryption key
//    encryptDatabase.key = "TODO: Your encryption key here";
//    options.encryptDatabase = encryptDatabase;

    // If you previously enrolled the templates into a PostgreSQL database, then use POSTGRESQL instead
//    options.dbms = Trueface::DatabaseManagementSystem::POSTGRESQL;

    // Since we know we will use the face detector and face recognizer,
    // we can choose to initialize these modules in the SDK constructor instead of using lazy initialization
    Trueface::InitializeModule initializeModule;
    initializeModule.faceDetector = true;
    initializeModule.faceRecognizer = true;
    options.initializeModule = initializeModule;

    Trueface::SDK tfSdk(options);
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

    if (retcode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to create database connection\n";
        return -1;
    }

    // Load the existing collection into memory.
    retcode = tfSdk.createLoadCollection(collectionName);
    if (retcode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to create or load collection\n";
        return -1;
    }

    // Use image of Brad Pitt as probe image
    retcode = tfSdk.setImage("../../images/brad_pitt_1.jpg");
    if (retcode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to set the image\n";
        return -1;
    }

    // Generate a template as the probe
    // We do not need to ensure that the probe template is high quality.
    // Only the enrollment templates must be high quality.
    Trueface::Faceprint probeFaceprint;
    bool foundFace;
    retcode = tfSdk.getLargestFaceFeatureVector(probeFaceprint, foundFace);
    if (retcode != Trueface::ErrorCode::NO_ERROR || !foundFace) {
        std::cout << "Unable to generate template or find face in image\n";
        return -1;
    }

    // Run identify function
    Trueface::Candidate candidate;
    bool found;

    // TODO: Select a threshold for your application using the ROC curves
    // https://docs.trueface.ai/ROC-Curves-d47d2730cf0a44afacb39aae0ed1b45a
    const float threshold = 0.4;
    retcode = tfSdk.identifyTopCandidate(probeFaceprint, candidate, found, threshold);

    if (retcode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "There was an error with the call to identify" << std::endl;
        return -1;
    }

    if (!found) {
        std::cout << "Unable to find match" << std::endl;
        return -1;
    }

    std::cout << "Identity found: " << candidate.identity << std::endl;
    std::cout << "Match Similarity: " << candidate.similarityMeasure << std::endl;
    std::cout << "Match Probability: " << candidate.matchProbability << std::endl;

    return 0;
}


