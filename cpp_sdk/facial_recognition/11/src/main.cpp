#include <iostream>
#include <vector>

#include "tf_sdk.h"
#include "tf_data_types.h"

int main() {
    // For a full list of configuration options, visit: https://reference.trueface.ai/cpp/dev/latest/usage/general.html#_CPPv4N8Trueface20ConfigurationOptionsE
    Trueface::ConfigurationOptions options;
    options.frModel = Trueface::FacialRecognitionModel::TFV5; 
    // Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file
    // TODO: If you have a NVIDIA gpu, then enable the enableGPU flag (you will require a GPU specific token for this).

    Trueface::SDK tfSdk (options);

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense("<LICENSE_CODE>");
    if (!isValid) {
        std::cout << "Error: the provided license is invalid\n";
        return -1;
    }

    // Load the first image of Obama
    auto errorCode = tfSdk.setImage("../../../../images/obama/obama1.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: unable to read image\n";
        return -1;
    }

    // Generate a template from the first image
    Trueface::Faceprint faceprint1;
    bool found;
    errorCode = tfSdk.getLargestFaceFeatureVector(faceprint1, found);
    if (errorCode != Trueface::ErrorCode::NO_ERROR || !found) {
        std::cout << "Error: Unable to generate template\n";
        return -1;
    }

    // Load the second image of obama
    errorCode = tfSdk.setImage("../../../../images/obama/obama2.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: uanble to read image\n";
        return -1;
    }

    // Generate a template from the second image
    Trueface::Faceprint faceprint2;
    errorCode = tfSdk.getLargestFaceFeatureVector(faceprint2, found);
    if (errorCode != Trueface::ErrorCode::NO_ERROR || !found) {
        std::cout << "Error: Unable to generate template\n";
        return -1;
    }

    // Compare two images of Obama
    float matchProbabilitiy, similarityMeasure;
    errorCode = tfSdk.getSimilarity(faceprint1, faceprint2, matchProbabilitiy, similarityMeasure);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to generate similarity score\n";
        return -1;
    }

    std::cout << "Similarity between two Obama images: " << matchProbabilitiy << "\n";

    // Load image of armstrong
    errorCode = tfSdk.setImage("../../../../images/armstrong/armstrong1.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: uanble to read image\n";
        return -1;
    }

    // Generate a template from the second third
    Trueface::Faceprint faceprint3;
    errorCode = tfSdk.getLargestFaceFeatureVector(faceprint3, found);
    if (errorCode != Trueface::ErrorCode::NO_ERROR || !found) {
        std::cout << "Error: Unable to generate template\n";
        return -1;
    }

    // Compare the image of Obama to Armstrong
    errorCode = tfSdk.getSimilarity(faceprint1, faceprint3, matchProbabilitiy, similarityMeasure);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to generate similarity score\n";
        return -1;
    }

    std::cout << "Similarity between Obama and Armstrong: " << matchProbabilitiy << "\n";

    return 0;
}