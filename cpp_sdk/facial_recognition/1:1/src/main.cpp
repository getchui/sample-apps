#include <iostream>
#include <vector>

#include "tf_sdk.h"
#include "tf_data_types.h"

int main() {
    Trueface::SDK tfSdk;

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense("<LICENSE_CODE>");
    if (!isValid) {
        std::cout << "Error: the provided license is invalid\n";
        return -1;
    }

    // Load the first image of Obama
    auto errorCode = tfSdk.setImage("../../../images/obama/obama1.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: unable to read image\n";
        return -1;
    }

    // Generate a template from the first image
    std::vector<float> templ1;
    errorCode = tfSdk.getLargestFaceFeatureVector(templ1);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to generate template\n";
        return -1;
    }

    // Load the second image of obama
    errorCode = tfSdk.setImage("../../../images/obama/obama2.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: uanble to read image\n";
        return -1;
    }

    // Generate a template from the second image
    std::vector<float> templ2;
    errorCode = tfSdk.getLargestFaceFeatureVector(templ2);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to generate template\n";
        return -1;
    }

    // Compare two images of Obama
    float similarity;
    errorCode = tfSdk.getSimilarity(templ1, templ2, similarity);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to generate similarity score\n";
        return -1;
    }

    std::cout << "Similarity between two Obama images: " << similarity << "\n";

    // Load image of armstrong
    errorCode = tfSdk.setImage("../../../images/armstrong/armstrong1.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: uanble to read image\n";
        return -1;
    }

    // Generate a template from the second third
    std::vector<float> templ3;
    errorCode = tfSdk.getLargestFaceFeatureVector(templ3);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to generate template\n";
        return -1;
    }

    // Compare the image of Obama to Armstrong
    errorCode = tfSdk.getSimilarity(templ1, templ3, similarity);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to generate similarity score\n";
        return -1;
    }

    std::cout << "Similarity between Obama and Armstrong: " << similarity << "\n";

    return 0;
}