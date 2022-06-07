// Sample code: load two images, extract feature vectors and compare the similarity
// This sample is similar to face_recognition.cpp, and shows how to call setImage() with an image buffer pointer.
// First two images of the same identity are compared. Then the similarity score of two different identities is computed.

#include "tf_sdk.h"
#define STB_IMAGE_IMPLEMENTATION
#include "3rd_party_libs/stb_image.h"
#include <iostream>
#include <vector>

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
    GPUModuleOptions gpuOptions;
    gpuOptions.enableGPU = false; // TODO: Change this to true to enable GPU inference.
    gpuOptions.maxBatchSize = 4;
    gpuOptions.optBatchSize = 1;
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

    // Load the first image into memory using STB_image (for the sake of the demo).
    int width, height, channels;
    uint8_t* rgb_image = stbi_load("../../images/brad_pitt_1.jpg", &width, &height, &channels, 3);

    // Preprocess the first image
    TFImage img1, img2, img3;
    auto errorCode = tfSdk.preprocessImage(rgb_image, width, height, ColorCode::rgb, img1);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }
    // Calling preprocessImage creates a copy of the data, therefore we can deallocate the input buffer
    stbi_image_free(rgb_image);


    // Load the second image
    rgb_image = stbi_load("../../images/brad_pitt_2.jpg", &width, &height, &channels, 3);
    errorCode = tfSdk.preprocessImage(rgb_image, width, height, ColorCode::rgb, img2);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }
    // Calling preprocessImage creates a copy of the data, therefore we can deallocate the input buffer
    stbi_image_free(rgb_image);

    // Can instead use preprocessRgbImage to load the image, if the image is already in rgb format.
    rgb_image = stbi_load("../../images/tom_cruise_1.jpg", &width, &height, &channels, 3);
    errorCode = tfSdk.preprocessRgbImage(rgb_image, width, height, img3);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

    // Note, when calling this version of the function, the data is not copied, but referenced.
    // We therefore cannot deallocate the buffer until we are done using it.

    // Generate the face recognition feature vectors for the three images.
    bool foundFace;
    Faceprint faceprint1, faceprint2, faceprint3;
    errorCode = tfSdk.getLargestFaceFeatureVector(img1, faceprint1, foundFace);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }
    if (!foundFace) {
        std::cout << "Unable to find face in image" << std::endl;
        return 1;
    }

    errorCode = tfSdk.getLargestFaceFeatureVector(img2, faceprint2, foundFace);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }
    if (!foundFace) {
        std::cout << "Unable to find face in image" << std::endl;
        return 1;
    }

    errorCode = tfSdk.getLargestFaceFeatureVector(img3, faceprint3, foundFace);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        return 1;
    }

    if (!foundFace) {
        std::cout << "Unable to find face in image" << std::endl;
        return 1;
    }

    // Now that we are done with the image which was loaded using preprocessRgbBuffer, we can deallocate the buffer.
    stbi_image_free(rgb_image);

    // Compute the similarity between the two images of the same face
    float similarityScore;
    float matchProbability;
    errorCode = SDK::getSimilarity(faceprint1, faceprint2, matchProbability, similarityScore);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute similarity score" << std::endl;
        std::cout << errorCode << std::endl;
        return 1;
    }

    std::cout <<  "Similarity score of same identity images: " << similarityScore << std::endl;
    std::cout <<  "Match probability of same identity images: " << matchProbability * 100 << "%\n" << std::endl;

    // Compute the similarity between the images of different identities.
    errorCode = SDK::getSimilarity(faceprint1, faceprint3, matchProbability, similarityScore);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << errorCode << std::endl;
        std::cout << "Unable to compute similarity score" << std::endl;
        return 1;
    }

    std::cout <<  "Similarity score of two different identities: " << similarityScore << std::endl;
    std::cout <<  "Match probability of two different identities: " << matchProbability * 100 << "%" << std::endl;

    return 0;
}