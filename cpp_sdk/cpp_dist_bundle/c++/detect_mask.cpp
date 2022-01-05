// Sample code: load an image, detect the largest face and check whether the face has a mask on or not
// First image is of a person wearing a mask. Second image is of a person not wearing a mask.
// The probability that a mask is worn over the face is computed in both cases.

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

    // Load the mask image and detect largest face.
    std::cout << "Image with mask" << std::endl;
    Trueface::ErrorCode errorCode = tfSdk.setImage("../../images/mask.jpg");
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

    // Run mask detection
    Trueface::MaskLabel maskLabel;
    errorCode = tfSdk.detectMask(faceBoxAndLandmarks, maskLabel);

    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout<<"Error: could not run mask detection"<<std::endl;
        return 1;
    }

    if (maskLabel == Trueface::MaskLabel::MASK) {
        std::cout << "Mask detected" << std::endl;
    } else {
        std::cout << "No mask detected" << std::endl;
    }

    // Load the non mask image and detect largest face.
    std::cout << "Image without mask" << std::endl;
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

    // Run mask detection
    errorCode = tfSdk.detectMask(faceBoxAndLandmarks, maskLabel);

    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout<<"Error: could not run mask detection"<<std::endl;
        return 1;
    }

    if (maskLabel == Trueface::MaskLabel::MASK) {
        std::cout << "Mask detected" << std::endl;
    } else {
        std::cout << "No mask detected" << std::endl;
    }
}