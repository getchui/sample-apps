// Sample code: load an image, run object detection
// This sample app demonstrates how to run object detection on an image
// An image of a person on a bike is first loaded. Next object detection is run and the predicted labels are printed.

#include "tf_sdk.h"
#include <iostream>
#include <vector>

int main() {
    Trueface::ConfigurationOptions options;

    // Since we know we will use the object detector module
    // we can choose to initialize this module in the SDK constructor instead of using lazy initialization
    Trueface::InitializeModule initializeModule;
    initializeModule.objectDetector = true;
    options.initializeModule = initializeModule;

    Trueface::SDK tfSdk(options);
    // TODO: Either input your token in the CMakeLists.txt file, or insert it below directly
    bool valid = tfSdk.setLicense(TRUEFACE_TOKEN);

    if (!valid) {
        std::cout<<"Error: the provided license is invalid."<<std::endl;
        return 1;
    }

    // Load the image of the person on a bike
    Trueface::ErrorCode errorCode = tfSdk.setImage("../../images/person_on_bike.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout<<"Error: could not load the image"<<std::endl;
        return 1;
    }

    std::vector<Trueface::BoundingBox> boundingBoxes;

    // Run object detection
    tfSdk.detectObjects(boundingBoxes);

    for (auto bbox : boundingBoxes) {
        // Convert the label to a string
        std::string label = tfSdk.getObjectLabelString(bbox.label);
        // Print out image label and probability
        std::cout << "Detected " << label << " with probability: " << bbox.probability << std::endl;
    }

    return 0;
}