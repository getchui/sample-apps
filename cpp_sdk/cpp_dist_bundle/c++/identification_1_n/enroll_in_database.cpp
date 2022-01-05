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

int main() {
    // For a full list of configuration options, visit: https://reference.trueface.ai/cpp/dev/latest/usage/general.html#_CPPv4N8Trueface20ConfigurationOptionsE
    Trueface::ConfigurationOptions options;
    options.frModel = Trueface::FacialRecognitionModel::TFV5;
    // Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file
    options.dbms = Trueface::DatabaseManagementSystem::SQLITE; // Save the templates in an SQLITE database
    // TODO: If you have a NVIDIA gpu, then enable the enableGPU flag (you will require a GPU specific token for this).
//    options.gpuOptions = true;

    // To enable database encryption...
//    Trueface::EncryptDatabase encryptDatabase;
//    encryptDatabase.enableEncryption = true;
//    // TODO: Replace with your own encryption key
//    encryptDatabase.key = "TODO: Your encryption key here";
//    options.encryptDatabase = encryptDatabase;
//

    // You can alternatively save the templates into a PostgreSQL database
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

    // Create the database
    auto retcode = tfSdk.createDatabaseConnection(databaseName);

    // If using the POSTGRESQL backend, then it might look something like this...
//    auto retcode = tfSdk.createDatabaseConnection("host=localhost port=5432 dbname=my_database user=postgres password=admin");

    if (retcode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to create database connection\n";
        return -1;
    }

    // Create a new collection, or if a collection with the given name already exists in the database, load the collection into memory.
    retcode = tfSdk.createLoadCollection(collectionName);
    if (retcode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to create or load collection\n";
        return -1;
    }

    // Since our collection is empty, lets populate the collection with some identities
    std::vector<std::pair<std::string, std::string>> identitiesVec = {
            {"../../images/brad_pitt_2.jpg", "Brad Pitt"},
            {"../../images/brad_pitt_3.jpg", "Brad Pitt"}, // Can add the same identity more than once
            {"../../images/tom_cruise_1.jpg", "Tom Cruise"}
    };

    for (const auto& identity: identitiesVec) {
        std::cout << "Processing image: " << identity.first << " with identity: " << identity.second << std::endl;
        // Start by setting the image
        retcode = tfSdk.setImage(identity.first);
        if (retcode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Unable to set the image\n";
            return -1;
        }

        // Detect the largest face in the image
        Trueface::FaceBoxAndLandmarks faceBoxAndLandmarks;
        bool faceDetected;
        auto errorCode = tfSdk.detectLargestFace(faceBoxAndLandmarks, faceDetected);
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "There was an error with the call to detectLargestFace\n";
            continue;
        }

        if (!faceDetected) {
            std::cout << "Unable to detect face\n";
            continue;
        }

        // We want to only enroll high quality images into the database / collection
        // For more information, refer to the section titled "Selecting the Best Enrollment Images"
        // https://reference.trueface.ai/cpp/dev/latest/usage/identification.html

        // Therefore, ensure that the face height is at least 100px
        auto faceHeight = faceBoxAndLandmarks.bottomRight.y - faceBoxAndLandmarks.topLeft.y;
        std::cout << "Face height: " << faceHeight << std::endl;
        if (faceHeight < 100) {
            std::cout << "The face is too small in the image for a high quality enrollment." << std::endl;
            continue;
        }

        // Get the aligned face chip so that we can compute the image quality
        uint8_t alignedImage[37632];
        errorCode = tfSdk.extractAlignedFace(faceBoxAndLandmarks, alignedImage);
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "There was an error extracting the aligned face\n";
            continue;
        }

        float quality;
        errorCode = tfSdk.estimateFaceImageQuality(alignedImage, quality);
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Unable to compute image quality\n";
            continue;
        }

        // Ensure the image quality is above a threshold
        // Once again, we only want to enroll only high quality images into our collection
        std::cout << "Face quality: " << quality << std::endl;
        if (quality < 0.999) {
            std::cout << "Please choose a higher quality enrollment image\n";
            return -1;
        }

        // We can check the orientation of the head and ensure that it is facing forward
        // To see the effect of yaw and pitch on the match score, refer to: https://reference.trueface.ai/cpp/dev/latest/usage/face.html#_CPPv4N8Trueface3SDK23estimateHeadOrientationERK19FaceBoxAndLandmarksRfRfRf
        float yaw, pitch, roll;
        errorCode = tfSdk.estimateHeadOrientation(faceBoxAndLandmarks, yaw, pitch, roll);
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Unable to compute head orientation\n";
            return -1;
        }

        float yawDeg = yaw * 180 / 3.14;
        float pitchDeg = pitch * 180 / 3.14;

        // Ensure the head is approximately neutral
        if (std::abs(yawDeg) > 50) {
            std::cout << "Enrollment image has too extreme a yaw: " << yawDeg <<
                      " deg. Please choose a higher quality enrollment image." << std::endl;
            return -1;
        }

        if (std::abs(pitchDeg) > 35) {
            std::cout << "Enrollment image has too extreme a pitch: " << pitchDeg <<
                      " deg. Please choose a higher quality enrollment image." << std::endl;
            return -1;
        }

        // Finally, ensure the user is not wearing a mask
        Trueface::MaskLabel masklabel;
        errorCode = tfSdk.detectMask(faceBoxAndLandmarks, masklabel);
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Error: Unable to compute mask score" << std::endl;
            return -1;
        }

        if (masklabel == Trueface::MaskLabel::MASK) {
            std::cout << "Please choose an image without a mask for enrollment." << std::endl;
            return -1;
        }

        // Generate the enrollment template
        Trueface::Faceprint enrollmentFaceprint;
        errorCode = tfSdk.getFaceFeatureVector(alignedImage, enrollmentFaceprint);
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Error: Unable to generate template\n";
            continue;
        }

        // Enroll the template into the collection
        std::string UUID;
        retcode = tfSdk.enrollFaceprint(enrollmentFaceprint, identity.second, UUID);
        if (retcode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Unable to enroll template\n";
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
    if (retcode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to get collection names!" << std::endl;
        return -1;
    }

    for (const auto& collectionName: collectionNames) {
        // For each collection in our database, get the metadata
        Trueface::CollectionMetadata metadata;
        retcode = tfSdk.getCollectionMetadata(collectionName, metadata);
        if (retcode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Unable to get collection metadata!" << std::endl;
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
        if (retcode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Unable to get collection identities!" << std::endl;
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


