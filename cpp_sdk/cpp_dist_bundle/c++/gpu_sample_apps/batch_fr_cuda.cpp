// Sample code: Using the GPU/CUDA backend extract facial feature vectors for a batch of face chips.

#include "tf_sdk.h"
#include <iostream>
#include <vector>

using namespace Trueface;

int main() {
    // Start by specifying the configuration options to be used.
    // Can choose to use default configuration options if preferred by calling the default SDK
    // constructor. Learn more about configuration options here:
    // https://reference.trueface.ai/cpp/dev/latest/usage/general.html
    ConfigurationOptions options;
    // The face recognition model to use. TFV5_2 balances accuracy and speed.
    options.frModel = FacialRecognitionModel::TFV5_2;
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
    // By default, the SDK uses lazy initialization, meaning modules are only initialized when they
    // are first used (on first inference). This is done so that modules which are not used do not
    // load their models into memory, and hence do not utilize memory. The downside to this is that
    // the first inference will be much slower as the model file is being decrypted and loaded into
    // memory. Therefore, if you know you will use a module, choose to pre-initialize the module,
    // which reads the model file into memory in the SDK constructor.
    InitializeModule initializeModule;
    initializeModule.faceDetector = true;
    initializeModule.faceRecognizer = true;
    options.initializeModule = initializeModule;

    // Options for enabling GPU
    // We will disable GPU inference, but you can easily enable it by modifying the following
    // options Note, you may require a specific GPU enabled token in order to enable GPU inference.
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
    options.gpuOptions.objectDetectorGPUOptions = gpuOptions;

    SDK tfSdk(options);

    // TODO: Either input your token in the CMakeLists.txt file, or insert it below directly
    bool valid = tfSdk.setLicense(TRUEFACE_TOKEN);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        return 1;
    }

    // Create vector to store the face chips
    std::vector<TFFacechip> facechips;

    // Run face detection on the 3 images in serial, then add the face chips the vector which we
    // allocated.
    for (int i = 0; i < 3; i++) {
        TFImage img;
        ErrorCode errorCode =
            tfSdk.preprocessImage("../../images/brad_pitt_" + std::to_string(i + 1) + ".jpg", img);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << errorCode << std::endl;
            return 1;
        }

        FaceBoxAndLandmarks faceBoxAndLandmarks;
        bool found = false;
        errorCode = tfSdk.detectLargestFace(img, faceBoxAndLandmarks, found);

        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << errorCode << std::endl;
            return 1;
        }

        if (!found) {
            std::cout << "Unable to find face in image" << std::endl;
            return 1;
        }

        TFFacechip facechip;
        errorCode = tfSdk.extractAlignedFace(img, faceBoxAndLandmarks, facechip);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Unable to extract aligned face chip" << std::endl;
            std::cout << errorCode << std::endl;
            return 1;
        }

        facechips.push_back(facechip);
    }

    std::vector<MaskLabel> maskLabels;
    std::vector<float> maskScores;
    auto res = tfSdk.detectMasks(facechips, maskLabels, maskScores);
    if (res != ErrorCode::NO_ERROR) {
        std::cout << "Unable to run mask detection" << std::endl;
        return 1;
    }

    for (size_t idx = 0; idx < maskLabels.size(); ++idx) {
        auto &maskLabel = maskLabel.at(idx);
        auto &maskScore = maskScores.at(idx);

        if (maskLabel == MaskLabel::MASK) {
            std::cout << "Masked face detected with probability of " << 1.0 - maskScore
                      << std::endl;
        } else {
            std::cout << "Unmasked face detected with probability of " << maskScore << std::endl;
        }
    }

    // Generate face recognition templates for those 3 face chips in batch.
    // Batch template generation increases throughput.
    std::vector<Faceprint> faceprints;
    res = tfSdk.getFaceFeatureVectors(facechips, faceprints);
    if (res != ErrorCode::NO_ERROR) {
        std::cout << "Unable to generate face feature vectors" << std::endl;
        return 1;
    }

    float similarity;
    float probability;

    // Run a few similarity queries.
    res = SDK::getSimilarity(faceprints[0], faceprints[1], probability, similarity);
    if (res != ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute sim score" << std::endl;
        return 1;
    }
    std::cout << "1st face - 2nd face, match probability: " << probability * 100
              << "%,  cosine similarity: " << similarity << std::endl;

    res = SDK::getSimilarity(faceprints[1], faceprints[2], probability, similarity);
    if (res != ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute sim score" << std::endl;
        return 1;
    }
    std::cout << "2nd face - 3rd face, match probability: " << probability * 100
              << "%, cosine similarity: " << similarity << std::endl;

    return 0;
}
