// Sample code: load an image, detect the largest face and check whether the face is wearing eye glasses or not
// First image is of a person wearing eye glasses. Second image is of a person not wearing eye glasses.
// The probability that eye glasses of some type are worn over the face is computed in both cases.

#include "tf_sdk.h"
#include <iostream>

int main() {
    Trueface::SDK tfSdk;
    // TODO: Either input your token in the CMakeLists.txt file, or insert it below directly
    bool valid = tfSdk.setLicense(TRUEFACE_TOKEN);

    if (!valid) {
        std::cout<<"Error: the provided license is invalid."<<std::endl;
        return 1;
    }

    // Load the glasses image and detect largest face.
    std::cout << "Image with glasses:" << std::endl;
    Trueface::ErrorCode errorCode = tfSdk.setImage("../../images/glasses.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout<<"Error: could not load the image"<<std::endl;
        return 1;
    }

    Trueface::FaceBoxAndLandmarks faceBoxAndLandmarks;
    bool found = false;
    errorCode = tfSdk.detectLargestFace(faceBoxAndLandmarks, found);

    if (!found || errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout<<"Error: face not found";
        return 1;
    }

    // Run glasses detection
    Trueface::GlassesLabel glassesLabel;
    float glassesScore;
    errorCode = tfSdk.detectGlasses(faceBoxAndLandmarks, glassesLabel, glassesScore);

    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout<<"Error: could not run glasses detection"<<std::endl;
        return 1;
    }

    if (glassesLabel == Trueface::GlassesLabel::GLASSES) {
        std::cout << "Glasses detected" << std::endl;
    } else {
        std::cout << "No glasses detected" << std::endl;
    }

    // Load the non glasses image and detect largest face.
    std::cout << "Image with glasses" << std::endl;
    errorCode = tfSdk.setImage("../../images/headshot.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout<<"Error: could not load the image"<<std::endl;
        return 1;
    }

    errorCode = tfSdk.detectLargestFace(faceBoxAndLandmarks, found);

    if (!found || errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout<<"Error: face not found";
        return 1;
    }

    // Run glasses detection
    errorCode = tfSdk.detectGlasses(faceBoxAndLandmarks, glassesLabel, glassesScore);

    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout<<"Error: could not run glasses detection"<<std::endl;
        return 1;
    }

    if (glassesLabel == Trueface::GlassesLabel::GLASSES) {
        std::cout << "Glasses detected" << std::endl;
    } else {
        std::cout << "No glasses detected" << std::endl;
    }
}