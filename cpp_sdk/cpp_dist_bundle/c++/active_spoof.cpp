// Sample code: Demonstrates how to run active spoof.
// Active spoof works by analyzing the way a persons face changes as they move closer to a camera.
// The active spoof solution therefore required two images and expects the face a certain distance from the camera.
// In the far image, the face should be about 18 inches from the camera, while in the near image,
// the face should be 7-8 inches from the camera.

// In this sample app, we run spoof detection using both a real image pair and spoof attempt image pair.

#include "tf_sdk.h"
#include <iostream>
#include <vector>

// Use Trueface namespace to avoid typing it out in full.
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

    // If encryption is enabled, must provide an encryption key
    options.encryptDatabase.enableEncryption = false;
    options.encryptDatabase.key = "TODO: Your encryption key here";

    // Initialize module in SDK constructor.
    // By default, the SDK uses lazy initialization, meaning modules are only initialized when they are first used (on first inference).
    // This is done so that modules which are not used do not load their models into memory, and hence do not utilize memory.
    // The downside to this is that the first inference will be much slower as the model file is being decrypted and loaded into memory.
    // Therefore, if you know you will use a module, choose to pre-initialize the module, which reads the model file into memory in the SDK constructor.
    InitializeModule initializeModule;
    initializeModule.activeSpoof = true;
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
        std::cout<<"Error: the provided license is invalid."<<std::endl;
        return 1;
    }

    // Start by analyzing real images
    {
        // Load the far image. The face must be about 18 inches from the camera.
        TFImage img;
        ErrorCode errorCode = tfSdk.preprocessImage("../../images/far_shot_real_person.jpg", img);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << errorCode << std::endl;
            return 1;
        }

        // Next, we need to detect if there is a face in the image
        bool found;
        FaceBoxAndLandmarks fb;

        auto ret = tfSdk.detectLargestFace(img, fb, found);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << errorCode << std::endl;
            return 1;
        }

        if (!found) {
            std::cout << "No face found in real image, far shot" << std::endl;
            return 1;
        }

        // Now, we need to check that the image meets our size criteria.
        // Be sure to check the return value from this function
        ret = tfSdk.checkSpoofImageFaceSize(img, fb, ActiveSpoofStage::FAR);
        if (ret == ErrorCode::FACE_TOO_FAR) {
            std::cout << "Real face, far shot: Face too far!" << std::endl;
            return 1;
        } else if (ret == ErrorCode::FACE_TOO_CLOSE) {
            std::cout << "Real face, far shot: Face too close!" << std::endl;
            return 1;
        } else if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error with real face, far shot!" << std::endl;
            std::cout << errorCode << std::endl;
            return 1;
        }

        // Now, we need to compute the 106 facial landmarks for the face
        Landmarks farFaceLandmarks{};
        ret = tfSdk.getFaceLandmarks(img, fb, farFaceLandmarks);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Unable to compute facial landmarks for real face, far shot!" << std::endl;
            std::cout << errorCode << std::endl;
            return 1;
        }

        // Finally, we can compute a face recognition template for the face,
        // and later use it to ensure the two active spoof images are from the same identity.
        Faceprint farFaceprint;
        ret = tfSdk.getFaceFeatureVector(img, fb, farFaceprint);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error generating the faceprint for real face, far shot!" << std::endl;
            return 1;
        }

        // Now at this point we can repeat all the above steps, but now for the near shot face image.

        errorCode = tfSdk.preprocessImage("../../images/near_shot_real_person.jpg", img);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << errorCode << std::endl;
            return 1;
        }

        ret = tfSdk.detectLargestFace(img, fb, found);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << errorCode << std::endl;
            return 1;
        }

        if (!found) {
            std::cout << "No face found in real image, far shot" << std::endl;
            return 1;
        }

        ret = tfSdk.checkSpoofImageFaceSize(img, fb,ActiveSpoofStage::NEAR); // Be sure to specify ActiveSpoofStage::NEAR this time.
        if (ret == ErrorCode::FACE_TOO_FAR) {
            std::cout << "Real face, near shot: Face too far!" << std::endl;
            return 1;
        } else if (ret == ErrorCode::FACE_TOO_CLOSE) {
            std::cout << "Real face, near shot: Face too close!" << std::endl;
            return 1;
        } else if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error with real face, far near!" << std::endl;
            std::cout << errorCode << std::endl;
            return 1;
        }

        // Now, we need to compute the 106 facial landmarks for the face
        Landmarks nearFaceLandmarks{};
        ret = tfSdk.getFaceLandmarks(img, fb, nearFaceLandmarks);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Unable to compute facial landmarks for real face, near shot!" << std::endl;
            std::cout << errorCode << std::endl;
            return 1;
        }

        Faceprint nearFaceprint;
        ret = tfSdk.getFaceFeatureVector(img, fb, nearFaceprint);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error generating the faceprint for real face, near shot!" << std::endl;
            std::cout << errorCode << std::endl;
            return 1;
        }

        // Finally, we can run the spoof function

        float spoofScore;
        SpoofLabel spoofLabel;
        ret = tfSdk.detectActiveSpoof(nearFaceLandmarks, farFaceLandmarks, spoofScore, spoofLabel);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Unable to compute active spoof!" << std::endl;
            std::cout << errorCode << std::endl;
            return 1;
        }

        std::cout << "Printing results for real image: " << std::endl;

        if (spoofLabel == SpoofLabel::FAKE) {
            std::cout << "SPOOF RESULTS: Spoof attempt detected!\n" << std::endl;
        } else {
            // Finally, as a last step, we can compare the two face recognition templates to ensure they are the same identity
            float matchProb, simScore;
            SDK::getSimilarity(nearFaceprint, farFaceprint, matchProb, simScore);
            if (simScore < 0.3) {
                std::cout << "SPOOF RESULTS: Image is real, but the images are not of the same identity!\n" << std::endl;
            } else {
                std::cout << "SPOOF RESULTS: Image is real, and both images are of the same identity\n" << std::endl;
            }
        }
    }

    // Now for the sake of the demo, let's repeat the entire process, but this time with two spoof attempt images

    {
        // Load the far image. The face must be about 18 inches from the camera.
        TFImage img;
        ErrorCode errorCode = tfSdk.preprocessImage("../../images/far_shot_fake_person.jpg", img);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << errorCode << std::endl;
            return 1;
        }

        // Next, we need to detect if there is a face in the image
        bool found;
        FaceBoxAndLandmarks fb;

        auto ret = tfSdk.detectLargestFace(img, fb, found);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << errorCode << std::endl;
            return 1;
        }

        if (!found) {
            std::cout << "No face found in real image, far shot" << std::endl;
            return 1;
        }

        // Now, we need to check that the image meets our size criteria.
        // Be sure to check the return value from this function
        ret = tfSdk.checkSpoofImageFaceSize(img, fb, ActiveSpoofStage::FAR);
        if (ret == ErrorCode::FACE_TOO_FAR) {
            std::cout << "Fake face, far shot: Face too far!" << std::endl;
            return 1;
        } else if (ret == ErrorCode::FACE_TOO_CLOSE) {
            std::cout << "Fake face, far shot: Face too close!" << std::endl;
            return 1;
        } else if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error with fake face, far shot!" << std::endl;
            std::cout << errorCode << std::endl;
            return 1;
        }

        // Now, we need to compute the 106 facial landmarks for the face
        Landmarks farFaceLandmarks{};
        ret = tfSdk.getFaceLandmarks(img, fb, farFaceLandmarks);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Unable to compute facial landmarks for fake face, far shot!" << std::endl;
            std::cout << errorCode << std::endl;
            return 1;
        }

        // Finally, we can compute a face recognition template for the face,
        // and later use it to ensure the two active spoof images are from the same identity.
        Faceprint farFaceprint;
        ret = tfSdk.getFaceFeatureVector(img, fb, farFaceprint);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error generating the faceprint for fake face, far shot!" << std::endl;
            std::cout << errorCode << std::endl;
            return 1;
        }

        // Now at this point we can repeat all the above steps, but now for the near shot face image.
        errorCode = tfSdk.preprocessImage("../../images/near_shot_fake_person.jpg", img);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << errorCode << std::endl;
            return 1;
        }

        ret = tfSdk.detectLargestFace(img, fb, found);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << errorCode << std::endl;
            return 1;
        }

        if (!found) {
            std::cout << "No face found in real image, far shot" << std::endl;
            return 1;
        }

        ret = tfSdk.checkSpoofImageFaceSize(img, fb, ActiveSpoofStage::NEAR); // Be sure to specify ActiveSpoofStage::NEAR this time.

        if (ret == ErrorCode::FACE_TOO_FAR) {
            std::cout << "Fake face, near shot: Face too far!" << std::endl;
            return 1;
        } else if (ret == ErrorCode::FACE_TOO_CLOSE) {
            std::cout << "Fake face, near shot: Face too close!" << std::endl;
            return 1;
        } else if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error with fake face, near shot!" << std::endl;
            std::cout << errorCode << std::endl;
            return 1;
        }

        // Now, we need to compute the 106 facial landmarks for the face
        Landmarks nearFaceLandmarks{};
        ret = tfSdk.getFaceLandmarks(img, fb, nearFaceLandmarks);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Unable to compute facial landmarks for fake face, near shot!" << std::endl;
            std::cout << errorCode << std::endl;
            return 1;
        }

        Faceprint nearFaceprint;
        ret = tfSdk.getFaceFeatureVector(img, fb, nearFaceprint);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error generating the faceprint for fake face, near shot!" << std::endl;
            std::cout << errorCode << std::endl;
            return 1;
        }

        // Finally, we can run the spoof function

        float spoofScore;
        SpoofLabel spoofLabel;
        ret = tfSdk.detectActiveSpoof(nearFaceLandmarks, farFaceLandmarks, spoofScore, spoofLabel);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Unable to compute active spoof!" << std::endl;
            std::cout << errorCode << std::endl;
            return 1;
        }

        std::cout << "Printing results for fake image: " << std::endl;

        if (spoofLabel == SpoofLabel::FAKE) {
            std::cout << "SPOOF RESULTS: Spoof attempt detected!\n" << std::endl;
        } else {
            // Finally, as a last step, we can compare the two face recognition templates to ensure they are the same identity
            float matchProb, simScore;
            SDK::getSimilarity(nearFaceprint, farFaceprint, matchProb, simScore);
            if (simScore < 0.3) {
                std::cout << "SPOOF RESULTS: Image is real, but the images are not of the same identity!\n" << std::endl;
            } else {
                std::cout << "SPOOF RESULTS: Image is real, and both images are of the same identity\n" << std::endl;
            }
        }
    }
}