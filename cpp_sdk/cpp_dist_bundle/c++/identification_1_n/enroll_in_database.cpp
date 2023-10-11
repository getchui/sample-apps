// Sample code: Generate face recognition templates for images and then enroll them into a collection.

// This sample app demonstrates how you can enroll face recognition templates or Faceprints into a collection on disk.
// First, we create a database and create a new collection within that database.
// Next, we generate face recognition templates and enroll those templates into the collection.
// Note, after running this sample app, you can run the identification_1_n sample app.

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
    options.smallestFaceHeight = 80; // Filter out faces smaller than 80 pixels as we want to ensure we only enroll high quality images
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
    initializeModule.faceOrientationDetector = true;
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
    options.gpuOptions.faceOrientationDetectorGPUOptions = moduleOptions;


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

    // Create the database
    auto retcode = tfSdk.createDatabaseConnection(databaseName);

    // If using the POSTGRESQL backend, then it might look something like this...
//    auto retcode = tfSdk.createDatabaseConnection("host=localhost port=5432 dbname=my_database user=postgres password=admin");

    if (retcode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to create database connection\n";
        std::cout << retcode << std::endl;
        return -1;
    }

    // Create a new collection, or if a collection with the given name already exists in the database, load the collection into memory.
    retcode = tfSdk.createLoadCollection(collectionName);
    if (retcode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to create or load collection\n";
        std::cout << retcode << std::endl;
        return -1;
    }

    // Since our collection is empty, lets populate the collection with some identities
    std::vector<std::pair<std::string, std::string>> identitiesVec = {
            {"../../images/brad_pitt_1.jpg", "Brad Pitt"},
            {"../../images/brad_pitt_2.jpg", "Brad Pitt"}, // Can add the same identity more than once
            {"../../images/tom_cruise_1.jpg", "Tom Cruise"}
    };

    for (const auto& identity: identitiesVec) {
        std::cout << "Processing image: " << identity.first << " with identity: " << identity.second << std::endl;
        // Start by setting the image
        TFImage img;
        retcode = tfSdk.preprocessImage(identity.first, img);
        if (retcode != ErrorCode::NO_ERROR) {
            std::cout << retcode << std::endl;
            return -1;
        }

        // Since we are enrolling images from disk, there is a possibility that the images may be oriented incorrectly.
        // Therefore, run the image orientation detector and rotate the images appropriately.
        RotateFlags imageRotation;
        retcode = tfSdk.getFaceImageRotation(img, imageRotation);
        if (retcode != ErrorCode::NO_ERROR) {
            std::cout << "There was an error computing the image rotation" << std::endl;
            std::cout << retcode << std::endl;
            continue;
        }

        // Rotate the image appropriately
        img->rotate(imageRotation);

        // Detect the largest face in the image
        FaceBoxAndLandmarks faceBoxAndLandmarks;
        bool faceDetected;
        auto errorCode = tfSdk.detectLargestFace(img, faceBoxAndLandmarks, faceDetected);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "There was an error with the call to detectLargestFace\n";
            std::cout << errorCode << std::endl;
            continue;
        }

        if (!faceDetected) {
            std::cout << "Unable to detect face\n";
            continue;
        }

        // We want to only enroll high quality images into the database / collection
        // For more information, refer to the section titled "Selecting the Best Enrollment Images"
        // https://reference.trueface.ai/cpp/dev/latest/usage/identification.html

        // Ensure that the image is not overly bright or dark, and that the exposure if good for face recognition
        FaceImageQuality quality;
        float percentImageBright, percentImageDark, percentFaceBright;
        errorCode = tfSdk.checkFaceImageExposure(img, faceBoxAndLandmarks, quality, percentImageBright, percentImageDark, percentFaceBright);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to run exposure check on the image" << std::endl;
            continue;
        }

        if (quality != FaceImageQuality::GOOD) {
            std::cout << "The image has suboptimal exposure for enrollment" << std::endl;
            continue;
        }

        // Get the aligned face chip so that we can compute the face blur
        TFFacechip facechip;
        errorCode = tfSdk.extractAlignedFace(img, faceBoxAndLandmarks, facechip);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "There was an error extracting the aligned face\n";
            std::cout << errorCode << std::endl;
            continue;
        }

        // Compute the face image blur and ensure it is not too blurry for face recognition
        float score{};
        errorCode = tfSdk.detectFaceImageBlur(facechip, quality, score);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to compute image blur\n";
            std::cout << errorCode << std::endl;
            continue;
        }

        if (quality != FaceImageQuality::GOOD) {
            std::cout << "The face image is too blurry for enrollment, skipping" << std::endl;
            continue;
        }

        // We can check the orientation of the head and ensure that it is facing forward
        // To see the effect of yaw and pitch on the match score, refer to: https://reference.trueface.ai/cpp/dev/latest/usage/face.html#_CPPv4N8Trueface3SDK23estimateHeadOrientationERK19FaceBoxAndLandmarksRfRfRf
        Landmarks landmarks;
        errorCode = tfSdk.getFaceLandmarks(img, faceBoxAndLandmarks, landmarks);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to get head landmarks\n";
            std::cout << errorCode << std::endl;
            return -1;
        }

        float yaw, pitch, roll;
        std::array<double, 3> rotationVec, translationVec;
        errorCode = tfSdk.estimateHeadOrientation(img, faceBoxAndLandmarks, landmarks, yaw, pitch, roll, rotationVec, translationVec);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to compute head orientation\n";
            std::cout << errorCode << std::endl;
            return -1;
        }

        float yawDeg = yaw * 180 / 3.14;
        float pitchDeg = pitch * 180 / 3.14;

        // Ensure the head is approximately neutral
        if (std::abs(yawDeg) > 30) {
            std::cout << "Enrollment image has too extreme a yaw: " << yawDeg <<
                      " deg. Please choose a higher quality enrollment image." << std::endl;
            return -1;
        }

        if (std::abs(pitchDeg) > 30) {
            std::cout << "Enrollment image has too extreme a pitch: " << pitchDeg <<
                      " deg. Please choose a higher quality enrollment image." << std::endl;
            return -1;
        }

        // Finally, ensure the user is not wearing a mask
        MaskLabel masklabel;
        float maskScore;
        errorCode = tfSdk.detectMask(img, faceBoxAndLandmarks, masklabel, maskScore);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Error: Unable to compute mask score" << std::endl;
            std::cout << errorCode << std::endl;
            return -1;
        }

        if (masklabel == MaskLabel::MASK) {
            std::cout << "Please choose an image without a mask for enrollment." << std::endl;
            return -1;
        }

        // Generate the enrollment template
        Faceprint enrollmentFaceprint;
        errorCode = tfSdk.getFaceFeatureVector(facechip, enrollmentFaceprint);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Error: Unable to generate template\n";
            std::cout << errorCode << std::endl;
            continue;
        }

        // Enroll the template into the collection
        std::string UUID;
        errorCode = tfSdk.enrollFaceprint(enrollmentFaceprint, identity.second, UUID);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to enroll template\n";
            std::cout << errorCode << std::endl;
            continue;
        }

        std::cout << "Success: Enrolled template with UUID: " << UUID << std::endl;
        // TODO: Can choose to store UUID for each faceprint which is enrolled into the collection
        // The UUID can later be used to delete the template from the collection

        std::cout << "----------------------------------\n" << std::endl;
    }

    // For the sake of the demonstration, print the information about all the collections
    std::vector<std::string> collectionNames;
    retcode = tfSdk.getCollectionNames(collectionNames);
    if (retcode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to get collection names!" << std::endl;
        std::cout << retcode << std::endl;
        return -1;
    }

    for (const auto& collectionName: collectionNames) {
        // For each collection in our database, get the metadata
        CollectionMetadata metadata;
        retcode = tfSdk.getCollectionMetadata(collectionName, metadata);
        if (retcode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to get collection metadata!" << std::endl;
            std::cout << retcode << std::endl;
            return -1;
        }

        std::cout << "Metadata for collection: " << metadata.collectionName << std::endl;
        std::cout << "Number of identities in collection: " << metadata.numIdentities << std::endl;
        std::cout << "Number of Faceprints in collection: " << metadata.numFaceprints << std::endl;
        std::cout << "Feature vector length bytes: " << metadata.featureVectorSizeBytes << std::endl;
        std::cout << "Face recognition model name: " << metadata.modelName << std::endl;

        // Print the first 10 identities & UUIDs in the collection
        std::unordered_multimap<std::string, std::string> identities;
        retcode = tfSdk.getCollectionIdentities(collectionName, identities);
        if (retcode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to get collection identities!" << std::endl;
            std::cout << retcode << std::endl;
            return -1;
        }

        int numIdentities = 0;
        for (const auto& identity: identities) {
            if (numIdentities++ >= 10) break;

            std::cout << "Identity: " << identity.first << ", UUID: " << identity.second << std::endl;
        }
    }

    // If we wanted to remove the identities from the collection...
//    for (const auto& identity: identitiesVec) {
//        unsigned int numFaceprintsRemoved;
//        retcode = tfSdk.removeByIdentity(identity.second, numFaceprintsRemoved);
//    }

    return 0;
}


