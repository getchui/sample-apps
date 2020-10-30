#include <iostream>

#include "tf_sdk.h"
#include "tf_data_types.h"
#include <fstream>
#include <istream>


std::vector<uint8_t> readToBuffer()

int main() {
    std::ifstream file("/home/cyrus/Downloads/tfv4_output.bin", std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cout << "Unable to read file" << std::endl;
        return -1;
    }

    for (int i = 0; i < buffer.size(); i+=4) {
        std::cout << *(reinterpret_cast<float*>((buffer.data() + i))) << " ";
    }
    std::cout << "\n";

    Trueface::SDK tfSdk;

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbW90aW9uIjp0cnVlLCJmciI6dHJ1ZSwiYnlwYXNzX2dwdV91dWlkIjp0cnVlLCJwYWNrYWdlX2lkIjpudWxsLCJleHBpcnlfZGF0ZSI6IjIwMjEtMDEtMDEiLCJncHVfdXVpZCI6W10sInRocmVhdF9kZXRlY3Rpb24iOnRydWUsIm1hY2hpbmVzIjoxLCJhbHByIjp0cnVlLCJuYW1lIjoiQ3lydXMgR1BVIiwidGtleSI6Im5ldyIsImV4cGlyeV90aW1lX3N0YW1wIjoxNjA5NDU5MjAwLjAsImF0dHJpYnV0ZXMiOnRydWUsInR5cGUiOiJvZmZsaW5lIiwiZW1haWwiOiJjeXJ1c0B0cnVlZmFjZS5haSJ9.2uQZTG_AXcHVXEFahbvkM8-gmosLPxjSSnbfEAz5gpY");
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
    Trueface::Faceprint faceprint1;
    errorCode = tfSdk.getLargestFaceFeatureVector(faceprint1);
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
    Trueface::Faceprint faceprint2;
    errorCode = tfSdk.getLargestFaceFeatureVector(faceprint2);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
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
    errorCode = tfSdk.setImage("../../../images/armstrong/armstrong1.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: uanble to read image\n";
        return -1;
    }

    // Generate a template from the second third
    Trueface::Faceprint faceprint3;
    errorCode = tfSdk.getLargestFaceFeatureVector(faceprint3);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
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