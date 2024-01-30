// Sample code: Using the GPU/CUDA backend demonstrate the use of batch inference.

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
    // For the sake of this sample app, will demonstrate how to set all the configuration options.
    // Some of them may not be relevant to this sample app.
    // The face recognition model to use. TFV5_2 balances accuracy and speed.
    options.frModel = FacialRecognitionModel::TFV5_2;
    // The face detection model
    options.fdModel = FaceDetectionModel::FAST;
    // The face detection filter.
    options.fdFilter = FaceDetectionFilter::BALANCED;
    // Smallest face height in pixels for the face detector.
    options.smallestFaceHeight = 40;
    // Use a global inference threadpool
    options.useGlobalInferenceThreadpool = true;
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
    initializeModule.faceOrientationDetector = true;
    initializeModule.landmarkDetector = true;
    initializeModule.faceTemplateQualityEstimator = true;
    initializeModule.maskDetector = true;
    initializeModule.blinkDetector = true;
    initializeModule.faceBlurDetector = true;
    options.initializeModule = initializeModule;

    // Options for enabling GPU
    // We will disable GPU inference, but you can easily enable it by modifying the following
    // options Note, you may require a specific GPU enabled token in order to enable GPU inference.
    options.gpuOptions = true;
    options.gpuOptions.deviceIndex = 0;

    GPUModuleOptions moduleOptions;
    moduleOptions.maxBatchSize = 4;
    moduleOptions.optBatchSize = 4;
    moduleOptions.maxWorkspaceSizeMb = 2000;
    moduleOptions.precision = Precision::FP16;

    options.gpuOptions.faceRecognizerGPUOptions = moduleOptions;
    options.gpuOptions.faceLandmarkDetectorGPUOptions = moduleOptions;
    options.gpuOptions.faceDetectorGPUOptions = moduleOptions;
    options.gpuOptions.maskDetectorGPUOptions = moduleOptions;
    options.gpuOptions.objectDetectorGPUOptions = moduleOptions;
    options.gpuOptions.faceOrientationDetectorGPUOptions = moduleOptions;
    options.gpuOptions.faceBlurDetectorGPUOptions = moduleOptions;
    options.gpuOptions.spoofDetectorGPUOptions = moduleOptions;
    options.gpuOptions.blinkDetectorGPUOptions = moduleOptions;
    options.gpuOptions.faceTemplateQualityEstimatorGPUOptions = moduleOptions;

    SDK tfSdk(options);

    // TODO: Either input your token in the CMakeLists.txt file, or insert it below directly
    bool valid = tfSdk.setLicense(TRUEFACE_TOKEN);

    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        return 1;
    }

    // Define the test data
    std::vector<std::string> testData {
        "brad_pitt_1.jpg",
        "brad_pitt_2.jpg",
        "brad_pitt_3.jpg",
        "brad_pitt_4.jpg",
    };

    // Preprocess the images
    std::vector<TFImage> tfImages;
    for (const auto& imgPath: testData) {
        TFImage img;
        const std::string fullPath = "../../images/" + imgPath;
        auto ret = tfSdk.preprocessImage(fullPath, img);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Unable to set image at path: " + fullPath << std::endl;
            std::cout << ret << std::endl;
            return -1;
        }

        tfImages.push_back(img);
    }

    // Run face image orientation detection on the images in batch
    std::vector<RotateFlags> rotateFlags;
    auto ret = tfSdk.getFaceImageRotations(tfImages, rotateFlags);
    if (ret != ErrorCode::NO_ERROR) {
        std::cout << "Unable to run get face image rotation" << std::endl;
        std::cout << ret << std::endl;
        return -1;
    }

    // Adjust the rotation of the images
    for (size_t i = 0; i < tfImages.size(); ++i) {
        tfImages[i]->rotate(rotateFlags[i]);
    }

    // Run face detection and extract the face chips
    std::vector<FaceBoxAndLandmarks> fbs;
    std::vector<TFFacechip> chips;

    for (const auto& tfImg: tfImages) {
        bool found;
        FaceBoxAndLandmarks fb;
        ret = tfSdk.detectLargestFace(tfImg, fb, found);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error running face detection" << std::endl;
            std::cout << ret << std::endl;
            return -1;
        }

        if (!found) {
            std::cout << "Unable to find face in image!" << std::endl;
            std::cout << "Skipping..." << std::endl;
            continue;
        }

        // Extract the face chip
        TFFacechip chip;
        ret = tfSdk.extractAlignedFace(tfImg, fb, chip);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Unable to extract aligned face chip" << std::endl;
            std::cout << ret << std::endl;
            return -1;
        }

        fbs.push_back(fb);
        chips.push_back(chip);
    }

    // Run 106 face landmark detection in batch
    std::vector<Landmarks> landmarksVec;
    ret = tfSdk.getFaceLandmarks(tfImages, fbs, landmarksVec);
    if (ret != ErrorCode::NO_ERROR) {
        std::cout << "Unable to get 106 face landmarks" << std::endl;
        std::cout << ret << std::endl;
        return -1;
    }

    // Run blink detection in batch
    std::vector<BlinkState> blinkStates;
    ret = tfSdk.detectBlinks(tfImages, landmarksVec, blinkStates);
    if (ret != ErrorCode::NO_ERROR) {
        std::cout << "Unable to run blink detection" << std::endl;
        std::cout << ret << std::endl;
        return -1;
    }

    // Run mask detection in batch
    std::vector<MaskLabel> maskLabels;
    std::vector<float> scores;
    ret = tfSdk.detectMasks(chips, maskLabels, scores);
    if (ret != ErrorCode::NO_ERROR) {
        std::cout << "Unable to run mask detection" << std::endl;
        std::cout << ret << std::endl;
        return -1;
    }

    // Run blur detection in batch
    std::vector<FaceImageQuality> qualities;
    ret = tfSdk.detectFaceImageBlurs(chips, qualities, scores);
    if (ret != ErrorCode::NO_ERROR) {
        std::cout << "Unable to run blur detection" << std::endl;
        std::cout << ret << std::endl;
        return -1;
    }

    // Run face template quality in batch
    std::vector<bool> areTemplateQualitiesGood;
    ret = tfSdk.estimateFaceTemplateQualities(chips, areTemplateQualitiesGood, scores);
    if (ret != ErrorCode::NO_ERROR) {
        std::cout << "Unable to run face template quality" << std::endl;
        std::cout << ret << std::endl;
        return -1;
    }

    // Run face recognition in batch
    std::vector<Faceprint> faceprints;
    ret = tfSdk.getFaceFeatureVectors(chips, faceprints);
    if (ret != ErrorCode::NO_ERROR) {
        std::cout << "Unable to generate face feature vectors" << std::endl;
        std::cout << ret << std::endl;
        return -1;
    }

    // TODO: Do something with the results

    return 0;
}
