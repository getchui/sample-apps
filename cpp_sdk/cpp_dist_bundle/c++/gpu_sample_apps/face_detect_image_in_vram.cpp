// Sample code: Using the GPU/CUDA backend detect largest face in an image already loaded in the graphics card's memory.

#include "tf_sdk.h"
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/cuda.hpp"

using namespace std;

int main() {
    Trueface::ConfigurationOptions options;
    options.frModel = Trueface::FacialRecognitionModel::TFV5;
    // Note, you will need to run the download script in /download_models to obtain the model file
    Trueface::GPUModuleOptions moduleOptions;
    options.gpuOptions = true;
    options.gpuOptions.faceDetectorGPUOptions.deviceIndex = 0;
    options.gpuOptions.faceRecognizerGPUOptions.deviceIndex = 0;

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

    vector<Trueface::Faceprint> faceprints1;
    vector<Trueface::Faceprint> faceprints2;

    {
        // using opencv to load the image in vram
        cv::Mat img = cv::imread("../../images/brad_pitt_1.jpg");
        cv::cuda::GpuMat mat;
        mat.upload(img);
        uchar* ptr = mat.data;

        // Set the image using the Trueface SDK directly from VRAM
        Trueface::ErrorCode errorCode = tfSdk.setImage(ptr, img.cols, img.rows, Trueface::ColorCode::bgr, mat.step);

        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout<<"Error: could not load the image"<<std::endl;
            return 1;
        }

        // Run face detection
        Trueface::FaceBoxAndLandmarks faceBoxAndLandmarks;
        bool found = false;
        errorCode = tfSdk.detectLargestFace(faceBoxAndLandmarks, found);
        if (!found || errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout<<"Error: could not detect a face"<<std::endl;
            return 1;
        } else {
            std::cout << "Face detected at following coordinates: " << std::endl;
            cout<<faceBoxAndLandmarks.topLeft.x<<endl;
            cout<<faceBoxAndLandmarks.topLeft.y<<endl;
            cout<<faceBoxAndLandmarks.bottomRight.x<<endl;
            cout<<faceBoxAndLandmarks.bottomRight.y<<endl;
        }

        // Generate a face recognition template for the detected face
        cv::cuda::GpuMat chipGpu(1, 112*112, CV_8UC3);
        errorCode = tfSdk.extractAlignedFace(faceBoxAndLandmarks, chipGpu.data);
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Unable to extract aligned face" << std::endl;
            return 1;
        }

        vector<uint8_t*> alignedFaceImages;
        alignedFaceImages.push_back(chipGpu.data);
        errorCode = tfSdk.getFaceFeatureVectors(alignedFaceImages, faceprints1);
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Unable to generate face feature vector" << std::endl;
            return 1;
        }
    }
    {
        // using opencv to load the image in vram
        cv::Mat img = cv::imread("../../images/brad_pitt_2.jpg");
        cv::cuda::GpuMat mat;
        mat.upload(img);
        uchar* ptr = mat.data;

        // Set the image using the Trueface SDK directly from VRAM
        Trueface::ErrorCode errorCode = tfSdk.setImage(ptr, img.cols, img.rows, Trueface::ColorCode::bgr, mat.step);

        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout<<"Error: could not load the image"<<std::endl;
            return 1;
        }

        // Run face detection
        Trueface::FaceBoxAndLandmarks faceBoxAndLandmarks;
        bool found = false;
        errorCode = tfSdk.detectLargestFace(faceBoxAndLandmarks, found);
        if (!found || errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout<<"Error: could not detect a face"<<std::endl;
            return 1;
        } else {
            std::cout << "Face detected at following coordinates: " << std::endl;
            cout<<faceBoxAndLandmarks.topLeft.x<<endl;
            cout<<faceBoxAndLandmarks.topLeft.y<<endl;
            cout<<faceBoxAndLandmarks.bottomRight.x<<endl;
            cout<<faceBoxAndLandmarks.bottomRight.y<<endl;
        }

        // Generate a face recognition template for the detected face
        cv::cuda::GpuMat chipGpu(1, 112*112, CV_8UC3);
        errorCode = tfSdk.extractAlignedFace(faceBoxAndLandmarks, chipGpu.data);
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Unable to extract aligned face" << std::endl;
            return 1;
        }

        vector<uint8_t*> alignedFaceImages;
        alignedFaceImages.push_back(chipGpu.data);

        errorCode = tfSdk.getFaceFeatureVectors(alignedFaceImages, faceprints2);
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Unable to generate face feature vector" << std::endl;
            return 1;
        }
    }

    // Compute the similarity score of the two faces
    float prob, cos;
    auto res = Trueface::SDK::getSimilarity(faceprints1[0], faceprints2[0], prob, cos);
    if (res != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute sim score" << std::endl;
        return 1;
    }
    cout<<"Probability: "<< prob <<endl;
    cout<<"Similarity: "<< cos <<endl;

    return 0;
}