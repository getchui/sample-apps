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
    ConfigurationOptions options;

    // Since we know we will use the active spoof module
    // we can choose to initialize this module in the SDK constructor instead of using lazy initialization
    Trueface::InitializeModule initializeModule;
    initializeModule.activeSpoof = true;
    options.initializeModule = initializeModule;

    // TODO: If you have a NVIDIA gpu, then enable the enableGPU flag (you will require a GPU specific token for this).
    // This will speed up face detection.
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
        ErrorCode errorCode = tfSdk.setImage("../../images/far_shot_real_person.jpg");
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Error: could not load the image" << std::endl;
            return 1;
        }

        // Next, we need to get the image properties.
        // These properties are used by the checkSpoofImageFaceSize() function.

        ImageProperties imageProperties;
        tfSdk.getImageProperties(imageProperties);

        // Next, we need to detect if there is a face in the image
        bool found;
        FaceBoxAndLandmarks fb;

        auto ret = tfSdk.detectLargestFace(fb, found);
        if (!found || ret != ErrorCode::NO_ERROR) {
            std::cout << "No face found in real image, far shot" << std::endl;
            return 0;
        }

        // Now, we need to check that the image meets our size criteria.
        // Be sure to check the return value from this function
        ret = tfSdk.checkSpoofImageFaceSize(fb, imageProperties, ActiveSpoofStage::FAR);
        if (ret == ErrorCode::FACE_TOO_FAR) {
            std::cout << "Real face, far shot: Face too far!" << std::endl;
            return 0;
        } else if (ret == ErrorCode::FACE_TOO_CLOSE) {
            std::cout << "Real face, far shot: Face too close!" << std::endl;
            return 0;
        } else if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error with real face, far shot!" << std::endl;
            return 0;
        }

        // Now, we need to compute the 106 facial landmarks for the face
        Landmarks farFaceLandmarks{};
        ret = tfSdk.getFaceLandmarks(fb, farFaceLandmarks);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Unable to compute facial landmarks for real face, far shot!" << std::endl;
            return 0;
        }

        // Finally, we can compute a face recognition template for the face,
        // and later use it to ensure the two active spoof images are from the same identity.
        Faceprint farFaceprint;
        ret = tfSdk.getFaceFeatureVector(fb, farFaceprint);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error generating the faceprint for real face, far shot!" << std::endl;
            return 0;
        }

        // Now at this point we can repeat all the above steps, but now for the near shot face image.

        errorCode = tfSdk.setImage("../../images/near_shot_real_person.jpg");
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Error: could not load the image" << std::endl;
            return 1;
        }

        tfSdk.getImageProperties(imageProperties);

        ret = tfSdk.detectLargestFace(fb, found);
        if (!found || ret != ErrorCode::NO_ERROR) {
            std::cout << "No face found in real image, near shot" << std::endl;
            return 0;
        }

        ret = tfSdk.checkSpoofImageFaceSize(fb, imageProperties,
                                            ActiveSpoofStage::NEAR); // Be sure to specify ActiveSpoofStage::NEAR this time.
        if (ret == ErrorCode::FACE_TOO_FAR) {
            std::cout << "Real face, near shot: Face too far!" << std::endl;
            return 0;
        } else if (ret == ErrorCode::FACE_TOO_CLOSE) {
            std::cout << "Real face, near shot: Face too close!" << std::endl;
            return 0;
        } else if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error with real face, near shot!" << std::endl;
            return 0;
        }

        // Now, we need to compute the 106 facial landmarks for the face
        Landmarks nearFaceLandmarks{};
        ret = tfSdk.getFaceLandmarks(fb, nearFaceLandmarks);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Unable to compute facial landmarks for real face, near shot!" << std::endl;
            return 0;
        }

        Faceprint nearFaceprint;
        ret = tfSdk.getFaceFeatureVector(fb, nearFaceprint);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error generating the faceprint for real face, near shot!" << std::endl;
            return 0;
        }

        // Finally, we can run the spoof function

        float spoofScore;
        SpoofLabel spoofLabel;
        ret = tfSdk.detectActiveSpoof(nearFaceLandmarks, farFaceLandmarks, spoofScore, spoofLabel);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Unable to compute active spoof!" << std::endl;
            return 0;
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
        ErrorCode errorCode = tfSdk.setImage("../../images/far_shot_fake_person.jpg");
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Error: could not load the image" << std::endl;
            return 1;
        }

        // Next, we need to get the image properties.
        // These properties are used by the checkSpoofImageFaceSize() function.

        ImageProperties imageProperties;
        tfSdk.getImageProperties(imageProperties);

        // Next, we need to detect if there is a face in the image
        bool found;
        FaceBoxAndLandmarks fb;

        auto ret = tfSdk.detectLargestFace(fb, found);
        if (!found || ret != ErrorCode::NO_ERROR) {
            std::cout << "No face found in fake image, far shot" << std::endl;
            return 0;
        }

        // Now, we need to check that the image meets our size criteria.
        // Be sure to check the return value from this function
        ret = tfSdk.checkSpoofImageFaceSize(fb, imageProperties, ActiveSpoofStage::FAR);
        if (ret == ErrorCode::FACE_TOO_FAR) {
            std::cout << "Fake face, far shot: Face too far!" << std::endl;
            return 0;
        } else if (ret == ErrorCode::FACE_TOO_CLOSE) {
            std::cout << "Fake face, far shot: Face too close!" << std::endl;
            return 0;
        } else if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error with fake face, far shot!" << std::endl;
            return 0;
        }

        // Now, we need to compute the 106 facial landmarks for the face
        Landmarks farFaceLandmarks{};
        ret = tfSdk.getFaceLandmarks(fb, farFaceLandmarks);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Unable to compute facial landmarks for fake face, far shot!" << std::endl;
            return 0;
        }

        // Finally, we can compute a face recognition template for the face,
        // and later use it to ensure the two active spoof images are from the same identity.
        Faceprint farFaceprint;
        ret = tfSdk.getFaceFeatureVector(fb, farFaceprint);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error generating the faceprint for fake face, far shot!" << std::endl;
            return 0;
        }

        // Now at this point we can repeat all the above steps, but now for the near shot face image.

        errorCode = tfSdk.setImage("../../images/near_shot_fake_person.jpg");
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "Error: could not load the image" << std::endl;
            return 1;
        }

        tfSdk.getImageProperties(imageProperties);

        ret = tfSdk.detectLargestFace(fb, found);
        if (!found || ret != ErrorCode::NO_ERROR) {
            std::cout << "No face found in fake image, near shot" << std::endl;
            return 0;
        }

        ret = tfSdk.checkSpoofImageFaceSize(fb, imageProperties,
                                            ActiveSpoofStage::NEAR); // Be sure to specify ActiveSpoofStage::NEAR this time.
        if (ret == ErrorCode::FACE_TOO_FAR) {
            std::cout << "Fake face, near shot: Face too far!" << std::endl;
            return 0;
        } else if (ret == ErrorCode::FACE_TOO_CLOSE) {
            std::cout << "Fake face, near shot: Face too close!" << std::endl;
            return 0;
        } else if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error with fake face, near shot!" << std::endl;
            return 0;
        }

        // Now, we need to compute the 106 facial landmarks for the face
        Landmarks nearFaceLandmarks{};
        ret = tfSdk.getFaceLandmarks(fb, nearFaceLandmarks);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Unable to compute facial landmarks for fake face, near shot!" << std::endl;
            return 0;
        }

        Faceprint nearFaceprint;
        ret = tfSdk.getFaceFeatureVector(fb, nearFaceprint);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "There was an error generating the faceprint for fake face, near shot!" << std::endl;
            return 0;
        }

        // Finally, we can run the spoof function

        float spoofScore;
        SpoofLabel spoofLabel;
        ret = tfSdk.detectActiveSpoof(nearFaceLandmarks, farFaceLandmarks, spoofScore, spoofLabel);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Unable to compute active spoof!" << std::endl;
            return 0;
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