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
    auto modelsPath = std::getenv("MODELS_PATH");
    if (modelsPath) {
        options.modelsPath = modelsPath;
    }
    // Enable vector compression to improve 1 to 1 comparison speed and 1 to N search speed.
    options.frVectorCompression = false;
    // Database management system for storage of biometric templates for 1 to N identification.
    options.dbms = DatabaseManagementSystem::SQLITE;

    // If encryption is enabled, must provide an encryption key
    options.encryptDatabase.enableEncryption = false;
    options.encryptDatabase.key = "TODO: Your encryption key here";

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
    options.gpuOptions = true; // Enable GPU inference
    options.gpuOptions.deviceIndex = 0;

    GPUModuleOptions gpuOptions;
    gpuOptions.maxBatchSize = 8;
    gpuOptions.optBatchSize = 3;
    gpuOptions.maxWorkspaceSizeMb = 2000;
    gpuOptions.precision = Precision::FP16;

    options.gpuOptions.faceRecognizerGPUOptions = gpuOptions;
    options.gpuOptions.faceDetectorGPUOptions = gpuOptions;
    options.gpuOptions.maskDetectorGPUOptions = gpuOptions;

    SDK tfSdk(options);

    // TODO: Either input your token in the CMakeLists.txt file, or insert it below directly
    bool valid = tfSdk.setLicense(TRUEFACE_TOKEN);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        return 1;
    }

    std::vector<Faceprint> faceprints1;
    std::vector<Faceprint> faceprints2;

    std::vector<std::string> imagePaths = {
            "../../images/brad_pitt_1.jpg",
            "../../images/brad_pitt_2.jpg"
    };

    std::vector<TFFacechip> facechips;

    for (const auto& imagePath: imagePaths) {
        // using opencv to load the image in vram
        cv::Mat img = cv::imread(imagePath);
        cv::cuda::GpuMat mat;
        mat.upload(img);
        uchar* ptr = mat.data;

        // Set the image using the Trueface SDK directly from VRAM
        TFImage gpuImg;
        ErrorCode errorCode = tfSdk.preprocessImage(ptr, img.cols, img.rows, ColorCode::bgr, gpuImg, mat.step);

        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << errorCode << std::endl;
            return -1;
        }

        // Detect the largest face, and then extract the aligned face, all while in GPU memory.
        FaceBoxAndLandmarks fb;
        bool found;
        errorCode = tfSdk.detectLargestFace(gpuImg, fb, found);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << errorCode << std::endl;
            return -1;
        }

        if (!found) {
            std::cout << "Unable to find face in image" << std::endl;
            return -1;
        }

        TFFacechip facechip;
        errorCode = tfSdk.extractAlignedFace(gpuImg, fb, facechip);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to extract aligned face" << std::endl;
            return 1;
        }

        facechips.push_back(facechip);
    }

    std::vector<Faceprint> faceprints;

    auto errorcode = tfSdk.getFaceFeatureVectors(facechips, faceprints);
    if (errorcode != ErrorCode::NO_ERROR) {
        std::cout << errorcode << std::endl;
        return -1;
    }

    // Compute the similarity score of the two faces
    float prob, cos;
    auto res = SDK::getSimilarity(faceprints[0], faceprints[1], prob, cos);
    if (res != ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute sim score" << std::endl;
        return 1;
    }

    std::cout << "Probability: "<< prob * 100 << "%" << std::endl;
    std::cout << "Similarity: "<< cos << std::endl;
    return 0;
}