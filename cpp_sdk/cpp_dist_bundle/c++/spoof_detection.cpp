// Sample code: sets an image and runs spoof detection on the image

// This sample demonstrates how to use the spoof detector to identify presentation attacks.
// First, the image is set. Next, the spood detection function is used to identify potential spoof.


#include "tf_sdk.h"
#include <iostream>

int main() {
    Trueface::SDK tfSdk;
    // TODO: Either input your token in the CMakeLists.txt file, or insert it below directly
    bool valid = tfSdk.setLicense(TRUEFACE_TOKEN);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        return 1;
    }

    // Load the real image
    Trueface::ErrorCode errorCode = tfSdk.setImage("../../images/real.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: could not load the image" << std::endl;
        return 1;
    }

    std::cout << "Running spoof detection with real image" << std::endl;
    Trueface::FaceBoxAndLandmarks faceBoxAndLandmarks;
    bool found = false;
    errorCode = tfSdk.detectLargestFace(faceBoxAndLandmarks, found);
    Trueface::SpoofLabel result;
    float spoofScore;
    if (found){
        errorCode = tfSdk.detectSpoof(faceBoxAndLandmarks, result, spoofScore);
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Error: could not run spoof detection" << std::endl;
            return 1;
        }
    }

    if (result == Trueface::SpoofLabel::REAL) {
        std::cout << "Real image detected" << std::endl;
    } else {
        std::cout << "Fake image detected" << std::endl;
    }

    // Load the fake image
    errorCode = tfSdk.setImage("../../images/fake.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: could not load the image" << std::endl;
        return 1;
    }

    std::cout << "Running spoof detection with fake image" << std::endl;
    errorCode = tfSdk.detectLargestFace(faceBoxAndLandmarks, found);
    if (found){
        errorCode = tfSdk.detectSpoof(faceBoxAndLandmarks, result, spoofScore);
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Error: could not run spoof detection" << std::endl;
            return 1;
        }
    }

    if (result == Trueface::SpoofLabel::REAL) {
        std::cout << "Real image detected" << std::endl;
    } else {
        std::cout << "Fake image detected" << std::endl;
    }

    return 0;
}