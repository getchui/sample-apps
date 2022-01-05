// Sample code: load images and determine if a person is blinking in the images

// This sample demonstrates how to use the detectBlink function. First, an image is loaded.
// Next, the blink scores are computed.


#include "tf_sdk.h"
#include <iostream>

int main() {
    Trueface::ConfigurationOptions options;

    // Since we know we will use the liveness
    // we can choose to initialize this module in the SDK constructor instead of using lazy initialization
    Trueface::InitializeModule initializeModule;
    initializeModule.liveness = true;
    options.initializeModule = initializeModule;

    Trueface::SDK tfSdk(options);
    // TODO: Either input your token in the CMakeLists.txt file, or insert it below directly
    bool valid = tfSdk.setLicense(TRUEFACE_TOKEN);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        return 1;
    }

    // Load the image with the eyes open
    Trueface::ErrorCode errorCode = tfSdk.setImage("../../images/open_eyes.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: could not load the image" << std::endl;
        return 1;
    }

    std::cout << "Running blink detection with open eye image" << std::endl;

    // Start by detecting the largest face in the image
    bool found;
    Trueface::FaceBoxAndLandmarks fb;

    errorCode = tfSdk.detectLargestFace(fb, found);
    if (!found || errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to detect face in image 1" << std::endl;
        return 0;
    }

    // Compute if the detected face has eyes open or closed

    Trueface::BlinkState blinkState;
    errorCode = tfSdk.detectBlink(fb, blinkState);
    if (errorCode == Trueface::ErrorCode::EXTREME_FACE_ANGLE) {
        std::cout << "The face angle is too extreme! Please ensure face image is forward facing!" << std::endl;
        return 0;
    } else if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute blink!" << std::endl;
        return 0;
    }

    // At this point, we can use the members of Trueface::BlinkState along with our own threshold to determine if the eyes are open or closed
    // Alternatively, we can use the pre-set thresholds by consulting Trueface::BlinkState.isLeftEyeClosed and Trueface::BlinkState.isRightEyeClosed

    std::cout << "Left eye score: " << blinkState.leftEyeScore << std::endl;
    std::cout << "Right eye score: " << blinkState.rightEyeScore << std::endl;
    std::cout << "Left eye aspect ratio: " << blinkState.leftEyeAspectRatio << std::endl;
    std::cout << "Right eye aspect ratio: " << blinkState.rightEyeAspectRatio << std::endl;

    if (blinkState.isLeftEyeClosed && blinkState.isRightEyeClosed) {
        std::cout << "PREDICTED RESULTS: Both eyes are closed!" << std::endl;
    } else {
        std::cout << "PREDICTED RESULTS:  The eyes are not closed!" << std::endl;
    }

    std::cout << "\nRunning blink detection with closed eye image" << std::endl;

    // Load the image with the eyes closed
    errorCode = tfSdk.setImage("../../images/closed_eyes.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: could not load the image" << std::endl;
        return 1;
    }

    errorCode = tfSdk.detectLargestFace(fb, found);
    if (!found || errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to detect face in image 2" << std::endl;
        return 0;
    }

    // Compute if the detected face has eyes open or closed

    errorCode = tfSdk.detectBlink(fb, blinkState);
    if (errorCode == Trueface::ErrorCode::EXTREME_FACE_ANGLE) {
        std::cout << "The face angle is too extreme! Please ensure face image is forward facing!" << std::endl;
        return 0;
    } else if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute blink!" << std::endl;
        return 0;
    }

    // At this point, we can use the members of Trueface::BlinkState along with our own threshold to determine if the eyes are open or closed
    // Alternatively, we can use the pre-set thresholds by consulting Trueface::BlinkState.isLeftEyeClosed and Trueface::BlinkState.isRightEyeClosed

    std::cout << "Left eye score: " << blinkState.leftEyeScore << std::endl;
    std::cout << "Right eye score: " << blinkState.rightEyeScore << std::endl;
    std::cout << "Left eye aspect ratio: " << blinkState.leftEyeAspectRatio << std::endl;
    std::cout << "Right eye aspect ratio: " << blinkState.rightEyeAspectRatio << std::endl;

    if (blinkState.isLeftEyeClosed && blinkState.isRightEyeClosed) {
        std::cout << "PREDICTED RESULTS: Both eyes are closed!" << std::endl;
    } else {
        std::cout << "PREDICTED RESULTS: The eyes are not closed!" << std::endl;
    }

    return 0;
}