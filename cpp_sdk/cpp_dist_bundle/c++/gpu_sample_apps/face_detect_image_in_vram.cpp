// Sample code: Using the GPU/CUDA backend detect largest face in an image already loaded in the graphics card's memory.

#include "tf_sdk.h"
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/cuda.hpp"

using namespace Trueface;

int main() {
    // Start by specifying the configuration options to be used.
    // Can choose to use default configuration options if preferred by calling the default SDK constructor.
    // Learn more about configuration options here: https://reference.trueface.ai/cpp/dev/latest/usage/general.html
    ConfigurationOptions options;
    // The face recognition model to use. Use the most accurate face recognition model.
    options.frModel = FacialRecognitionModel::TFV5;
    // The object detection model to use.
    options.objModel = ObjectDetectionModel::ACCURATE;
    // The face detection filter.
    options.fdFilter = FaceDetectionFilter::BALANCED;
    // Smallest face height in pixels for the face detector.
    options.smallestFaceHeight = 40;
    // The path specifying the directory where the model files have been downloaded
    options.modelsPath = "./";
    // Enable vector compression to improve 1 to 1 comparison speed and 1 to N search speed.
    options.frVectorCompression = false;
    // Database management system for storage of biometric templates for 1 to N identification.
    options.dbms = DatabaseManagementSystem::SQLITE;

    // Choose to encrypt the database
    EncryptDatabase encryptDatabase;
    encryptDatabase.enableEncryption = false; // TODO: To encrypt the database change this to true
    encryptDatabase.key = "TODO: Your encryption key here";
    options.encryptDatabase = encryptDatabase;

    // Initialize module in SDK constructor.
    // By default, the SDK uses lazy initialization, meaning modules are only initialized when they are first used (on first inference).
    // This is done so that modules which are not used do not load their models into memory, and hence do not utilize memory.
    // The downside to this is that the first inference will be much slower as the model file is being decrypted and loaded into memory.
    // Therefore, if you know you will use a module, choose to pre-initialize the module, which reads the model file into memory in the SDK constructor.
    InitializeModule initializeModule;
    initializeModule.faceDetector = true;
    initializeModule.faceRecognizer = true;
    options.initializeModule = initializeModule;

    // Options for enabling GPU
    // We will disable GPU inference, but you can easily enable it by modifying the following options
    // Note, you may require a specific GPU enabled token in order to enable GPU inference.
    GPUModuleOptions gpuOptions;
    gpuOptions.enableGPU = true;
    gpuOptions.maxBatchSize = 8;
    gpuOptions.optBatchSize = 3;
    gpuOptions.maxWorkspaceSizeMb = 2000;
    gpuOptions.deviceIndex = 0;
    gpuOptions.precision = Precision::FP16;

    options.gpuOptions.faceRecognizerGPUOptions = gpuOptions;
    options.gpuOptions.faceDetectorGPUOptions = gpuOptions;

    // Alternatively, can also do the following to enable GPU inference for all supported modules:
//    options.gpuOptions = true;

    SDK tfSdk(options);

    // TODO: Either input your token in the CMakeLists.txt file, or insert it below directly
    bool valid = tfSdk.setLicense(TRUEFACE_TOKEN);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        return 1;
    }

    std::vector<Faceprint> faceprints1;
    std::vector<Faceprint> faceprints2;

    {
        // using opencv to load the image in vram
        cv::Mat img = cv::imread("../../images/brad_pitt_1.jpg");
        cv::cuda::GpuMat mat;
        mat.upload(img);
        uchar* ptr = mat.data;

        // Set the image using the Trueface SDK directly from VRAM
        ErrorCode errorCode = tfSdk.setImage(ptr, img.cols, img.rows, ColorCode::bgr, mat.step);

        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Error: could not load the image" << std::endl;
            return 1;
        }

        // Run face detection
        FaceBoxAndLandmarks faceBoxAndLandmarks;
        bool found = false;
        errorCode = tfSdk.detectLargestFace(faceBoxAndLandmarks, found);
        if (!found || errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Error: could not detect a face" << std::endl;
            return 1;
        } else {
            std::cout << "Face detected at following coordinates: " << std::endl;
            std::cout <<faceBoxAndLandmarks.topLeft.x<< std::endl;
            std::cout <<faceBoxAndLandmarks.topLeft.y<< std::endl;
            std::cout <<faceBoxAndLandmarks.bottomRight.x<< std::endl;
            std::cout <<faceBoxAndLandmarks.bottomRight.y<< std::endl;
        }

        // Generate a face recognition template for the detected face
        cv::cuda::GpuMat chipGpu(1, 112*112, CV_8UC3);
        errorCode = tfSdk.extractAlignedFace(faceBoxAndLandmarks, chipGpu.data);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to extract aligned face" << std::endl;
            return 1;
        }

        std::vector<uint8_t*> alignedFaceImages;
        alignedFaceImages.push_back(chipGpu.data);
        errorCode = tfSdk.getFaceFeatureVectors(alignedFaceImages, faceprints1);
        if (errorCode != ErrorCode::NO_ERROR) {
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
        ErrorCode errorCode = tfSdk.setImage(ptr, img.cols, img.rows, ColorCode::bgr, mat.step);

        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Error: could not load the image" << std::endl;
            return 1;
        }

        // Run face detection
        FaceBoxAndLandmarks faceBoxAndLandmarks;
        bool found = false;
        errorCode = tfSdk.detectLargestFace(faceBoxAndLandmarks, found);
        if (!found || errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Error: could not detect a face" << std::endl;
            return 1;
        } else {
            std::cout << "Face detected at following coordinates: " << std::endl;
            std::cout <<faceBoxAndLandmarks.topLeft.x<< std::endl;
            std::cout <<faceBoxAndLandmarks.topLeft.y<< std::endl;
            std::cout <<faceBoxAndLandmarks.bottomRight.x<< std::endl;
            std::cout <<faceBoxAndLandmarks.bottomRight.y<< std::endl;
        }

        // Generate a face recognition template for the detected face
        cv::cuda::GpuMat chipGpu(1, 112*112, CV_8UC3);
        errorCode = tfSdk.extractAlignedFace(faceBoxAndLandmarks, chipGpu.data);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to extract aligned face" << std::endl;
            return 1;
        }

        std::vector<uint8_t*> alignedFaceImages;
        alignedFaceImages.push_back(chipGpu.data);

        errorCode = tfSdk.getFaceFeatureVectors(alignedFaceImages, faceprints2);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to generate face feature vector" << std::endl;
            return 1;
        }
    }

    // Compute the similarity score of the two faces
    float prob, cos;
    auto res = SDK::getSimilarity(faceprints1[0], faceprints2[0], prob, cos);
    if (res != ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute sim score" << std::endl;
        return 1;
    }
    std::cout << "Probability: "<< prob << std::endl;
    std::cout << "Similarity: "<< cos << std::endl;

    return 0;
}