// Sample code: Initialize the SDK using TFV5 model, load two images, extract feature vectors and compare the similarity
// First two images of the same identity are compared. Then the similarity score of two different identities is computed.

#include "tf_sdk.h"
#include <iostream>
#include <vector>

int main() {
    Trueface::ConfigurationOptions options;
    options.frModel = Trueface::FacialRecognitionModel::LITE_V2; // Use our most accurate model
    // Note, you will need to run the download script in /download_models to obtain the model file
    // TODO: If you have a NVIDIA gpu, then enable the enableGPU flag (you will require a GPU specific token for this).
//    options.gpuOptions = true;

    // Since we know we will use the face detector and face recognizer,
    // we can choose to initialize these modules in the SDK constructor instead of using lazy initialization
    Trueface::InitializeModule initializeModule;
    initializeModule.faceDetector = true;
    initializeModule.faceRecognizer = true;
    options.initializeModule = initializeModule;

    Trueface::SDK tfSdk(options);
    // TODO: Either input your token in the CMakeLists.txt file, or insert it below directly
    bool valid = tfSdk.setLicense(TRUEFACE_TOKEN);
    
    if (!valid) {
        std::cout<<"Error: the provided license is invalid."<<std::endl;
        return 1;
    }

    // Load the first image and extract the feature vector.
    Trueface::ErrorCode errorCode = tfSdk.setImage("../../images/brad_pitt_1.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout<<"Error: could not load the image"<<std::endl;
        return 1;
    }

    Trueface::Faceprint faceprint1;
    bool foundFace;
    errorCode = tfSdk.getLargestFaceFeatureVector(faceprint1, foundFace);
    if (errorCode != Trueface::ErrorCode::NO_ERROR || !foundFace) {
        std::cout<<"Error: Unable to detect face in image"<<std::endl;
        return 1;
    }

    // Load the second image and extract the feature vector.
    errorCode = tfSdk.setImage("../../images/brad_pitt_2.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout<<"Error: could not load the image 2"<<std::endl;
        return 1;
    }

    Trueface::Faceprint faceprint2;
    errorCode = tfSdk.getLargestFaceFeatureVector(faceprint2, foundFace);
    if (errorCode != Trueface::ErrorCode::NO_ERROR || !foundFace) {
        std::cout<<"Error: Unable to detect face in image"<<std::endl;
        return 1;
    }

    // Compute the similarity between the two face images.
    float similarityScore;
    float matchProbability;
    errorCode = Trueface::SDK::getSimilarity(faceprint1, faceprint2, matchProbability, similarityScore);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute similarity score" << std::endl;
        return 1;
    }

    std::cout<<"Similarity score of same identity images: "<<similarityScore<<std::endl;
    std::cout<<"Match probability of same identity images: "<<matchProbability<<std::endl;

    // Load the image of a different identity and extract the feature vector.
    errorCode = tfSdk.setImage("../../images/tom_cruise_1.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout<<"Error: could not load the image 2"<<std::endl;
        return 1;
    }

    Trueface::Faceprint faceprint3;
    errorCode = tfSdk.getLargestFaceFeatureVector(faceprint3, foundFace);
    if (errorCode != Trueface::ErrorCode::NO_ERROR || !foundFace) {
        std::cout<<"Error: Unable to detect face in image"<<std::endl;
        return 1;
    }

    // Compute the similarity between the images of different identities.
    errorCode = Trueface::SDK::getSimilarity(faceprint1, faceprint3, matchProbability, similarityScore);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute similarity score" << std::endl;
        return 1;
    }

    std::cout<<"Similarity score of two different identities: "<<similarityScore<<std::endl;
    std::cout<<"Match probability of two different identities: "<<matchProbability<<std::endl;

    return 0;
}