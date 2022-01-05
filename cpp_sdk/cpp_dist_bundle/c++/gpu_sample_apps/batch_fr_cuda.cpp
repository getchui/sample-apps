// Sample code: Using the GPU/CUDA backend extract facial feature vectors for a batch of face chips.

#include "tf_sdk.h"
#include <iostream>
#include <vector>

int main() {
    Trueface::ConfigurationOptions options;
    options.frModel = Trueface::FacialRecognitionModel::TFV5; // Use our most accurate face recognition model
    // Note, you will need to run the download script in /download_models to obtain the model file
    options.gpuOptions = true;
    options.gpuOptions.faceRecognizerGPUOptions.precision = Trueface::Precision::FP16;
    options.gpuOptions.faceRecognizerGPUOptions.maxBatchSize = 8;
    options.gpuOptions.faceRecognizerGPUOptions.optBatchSize = 3;

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
        std::cout << "Error: the provided license is invalid." << std::endl;
        return 1;
    }

    // Allocate a buffer to hold the face chips
    std::vector<uint8_t*> vecFaceChips;
    uint8_t faceChips[3][112][112][3];

    // Run face detection on the 3 images in serial, then add the face chips the vector which we allocated.
    for (int i=0; i<3; i++) {
        Trueface::ErrorCode errorCode = tfSdk.setImage("../../images/brad_pitt_"+std::to_string(i+1)+".jpg");
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout<<"Error: could not load the image"<<std::endl;
            return 1;
        }

        Trueface::FaceBoxAndLandmarks faceBoxAndLandmarks;
        bool found = false;
        errorCode = tfSdk.detectLargestFace(faceBoxAndLandmarks, found);
        if (!found || errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout<<"Error: could not detect a face"<<std::endl;
            return 1;
        }

        errorCode = tfSdk.extractAlignedFace(faceBoxAndLandmarks, &(faceChips[i][0][0][0]));
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Unable to extract aligned face chip" << std::endl;
            return 1;
        }
        vecFaceChips.push_back(&(faceChips[i][0][0][0]));
    }

    // Generate face recognition templates for those 3 face chips in batch.
    // Batch template generation increases throughput.
    std::vector<Trueface::Faceprint> faceprints;
    auto res = tfSdk.getFaceFeatureVectors(vecFaceChips, faceprints);
    if (res != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to generate face feature vectors" << std::endl;
        return 1;
    }

    float similarity;
    float probability;

    // Run a few similarity queries.
    res = Trueface::SDK::getSimilarity(faceprints[0], faceprints[1], probability, similarity);
    if (res != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute sim score" << std::endl;
        return 1;
    }
    std::cout<<"1st face - 2nd face, match probability: "<<probability
             <<" cosine similarity: "<<similarity<<std::endl;     

    res = Trueface::SDK::getSimilarity(faceprints[1], faceprints[2], probability, similarity);
    if (res != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute sim score" << std::endl;
        return 1;
    }
    std::cout<<"2nd face - 3rd face, match probability: "<<probability
             <<" cosine similarity: "<<similarity<<std::endl;     


    return 0;
}
