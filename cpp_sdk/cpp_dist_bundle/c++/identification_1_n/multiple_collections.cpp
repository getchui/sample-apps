// Sample code: Generate face recognition templates for images and then enroll them into two
// different collections, and then query each collection.

#include "tf_sdk.h"
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

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
    options.smallestFaceHeight = 80; // Filter out faces smaller than 80 pixels as we want to ensure
                                     // we only enroll high quality images
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
    initializeModule.faceOrientationDetector = true;
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
    options.gpuOptions.faceOrientationDetectorGPUOptions = moduleOptions;

    SDK tfSdk(options);
    // TODO: Either input your token in the CMakeLists.txt file, or insert it below directly
    bool valid = tfSdk.setLicense(TRUEFACE_TOKEN);

    if (valid == false) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        return 1;
    }

    // The name of our database, and the name of our collection in that database.
    const std::string databaseName = "multiple_collections.db";
    const std::string collection1 = "collection_1";
    const std::string collection2 = "collectioN_2";

    // Create the database
    auto retcode = tfSdk.createDatabaseConnection(databaseName);

    // If using the POSTGRESQL backend, then it might look something like this...
    //    auto retcode = tfSdk.createDatabaseConnection("host=localhost port=5432 dbname=my_database
    //    user=postgres password=admin");

    if (retcode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to create database connection\n";
        std::cout << retcode << std::endl;
        return -1;
    }

    // Create the two collections
    retcode = tfSdk.createCollection(collection1);
    if (retcode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to create collection: " + collection1 << std::endl;
        std::cout << retcode << std::endl;
        return -1;
    }

    retcode = tfSdk.createCollection(collection2);
    if (retcode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to create collection: " + collection2 << std::endl;
        std::cout << retcode << std::endl;
        return -1;
    }

    // Load the two collections into memory
    retcode = tfSdk.loadCollections({collection1, collection2});
    if (retcode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to load the collections into memory: " + collection2 << std::endl;
        std::cout << retcode << std::endl;
        return -1;
    }

    // Enroll Faceprint of Brad Pitt in collection 1
    {
        TFImage img;
        retcode = tfSdk.preprocessImage("../../images/brad_pitt_1.jpg", img);
        if (retcode != ErrorCode::NO_ERROR) {
            std::cout << retcode << std::endl;
            return -1;
        }

        // TODO: You should run additional quality checks before enrolling. See
        // enroll_in_database.cpp
        bool found;
        Faceprint fp;
        retcode = tfSdk.getLargestFaceFeatureVector(img, fp, found);
        if (!found || retcode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to generate feature vector from image" << std::endl;
            std::cout << retcode << std::endl;
            return -1;
        }

        // Enroll the Faceprint. Since we have multiple collections loaded we must specify the
        // collection name
        std::string UUID;
        retcode = tfSdk.enrollFaceprint(fp, "Brad Pitt", UUID, collection1);
        if (retcode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to enroll faceprint in collection: " << collection1 << std::endl;
            std::cout << retcode << std::endl;
            return -1;
        }
    }

    // Enroll Faceprint of Tom Cruise in collection 2
    {
        TFImage img;
        retcode = tfSdk.preprocessImage("../../images/tom_cruise_1.jpg", img);
        if (retcode != ErrorCode::NO_ERROR) {
            std::cout << retcode << std::endl;
            return -1;
        }

        // TODO: You should run additional quality checks before enrolling. See
        // enroll_in_database.cpp
        bool found;
        Faceprint fp;
        retcode = tfSdk.getLargestFaceFeatureVector(img, fp, found);
        if (!found || retcode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to generate feature vector from image" << std::endl;
            std::cout << retcode << std::endl;
            return -1;
        }

        // Enroll the Faceprint. Since we have multiple collections loaded we must specify the
        // collection name
        std::string UUID;
        retcode = tfSdk.enrollFaceprint(fp, "Tom Cruise", UUID, collection2);
        if (retcode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to enroll faceprint in collection: " << collection2 << std::endl;
            std::cout << retcode << std::endl;
            return -1;
        }
    }

    // Now run a search query in collection 1
    {
        TFImage img;
        retcode = tfSdk.preprocessImage("../../images/brad_pitt_2.jpg", img);
        if (retcode != ErrorCode::NO_ERROR) {
            std::cout << retcode << std::endl;
            return -1;
        }

        bool found;
        Faceprint fp;
        retcode = tfSdk.getLargestFaceFeatureVector(img, fp, found);
        if (!found || retcode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to generate feature vector from image" << std::endl;
            std::cout << retcode << std::endl;
            return -1;
        }

        // Query the collection. Since we have multiple collections loaded in memory, we must
        // specify the collection name
        Candidate candidate;
        retcode = tfSdk.identifyTopCandidate(fp, candidate, found, 0.4, collection1);
        if (retcode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to perform 1 to N search query" << std::endl;
            std::cout << retcode << std::endl;
            return -1;
        }

        if (found) {
            std::cout << "Found candidate in collection: " << collection1
                      << " with identity: " << candidate.identity
                      << " with match probability: " << candidate.matchProbability << std::endl;
        } else {
            std::cout << "Was not able to find match candidate in collection: " << collection1
                      << std::endl;
        }
    }

    // Now run a search query in collection 2
    {
        TFImage img;
        retcode = tfSdk.preprocessImage("../../images/tom_cruise_2.jpg", img);
        if (retcode != ErrorCode::NO_ERROR) {
            std::cout << retcode << std::endl;
            return -1;
        }

        bool found;
        Faceprint fp;
        retcode = tfSdk.getLargestFaceFeatureVector(img, fp, found);
        if (!found || retcode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to generate feature vector from image" << std::endl;
            std::cout << retcode << std::endl;
            return -1;
        }

        // Query the collection. Since we have multiple collections loaded in memory, we must
        // specify the collection name
        Candidate candidate;
        retcode = tfSdk.identifyTopCandidate(fp, candidate, found, 0.4, collection2);
        if (retcode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to perform 1 to N search query" << std::endl;
            std::cout << retcode << std::endl;
            return -1;
        }

        if (found) {
            std::cout << "Found candidate in collection: " << collection2
                      << " with identity: " << candidate.identity
                      << " with match probability: " << candidate.matchProbability << std::endl;
        } else {
            std::cout << "Was not able to find match candidate in collection: " << collection2
                      << std::endl;
        }
    }

    return 0;
}
