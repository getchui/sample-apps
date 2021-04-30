#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <thread>
#include <mutex>
#include <memory>
#include <atomic>
#include <chrono>
#include <cmath>

#include "tf_sdk.h"
#include "tf_data_types.h"

// Utility function for drawing the label on our image
void setLabel(cv::Mat& im, const std::string label, const cv::Point & oldOrigin, const cv::Scalar& color) {
    cv::Point origin(oldOrigin.x - 2, oldOrigin.y - 10);
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    // Can change scale and thickness to change label size
    const double scale = 0.8;
    const int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, font, scale, thickness, &baseline);
    cv::rectangle(im, origin + cv::Point(0, baseline), origin + cv::Point(text.width, -text.height), color, cv::FILLED);
    cv::putText(im, label, origin, font, scale, CV_RGB(0,0,0), thickness, cv::LINE_AA);
}


int main() {
    // TODO: Select a threshold for your application using the ROC curves
    // https://performance.trueface.ai/
    const float threshold = 0.3;

    // TODO: If you have a NVIDIA gpu, then enable the enableGPU flag (you will require a GPU specific token for this).
    Trueface::ConfigurationOptions options;
    options.frModel = Trueface::FacialRecognitionModel::TFV5; 
    // Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file

    options.dbms = Trueface::DatabaseManagementSystem::NONE; // The data will not persist after the app terminates using this backend option.
    options.smallestFaceHeight = 40; // https://reference.trueface.ai/cpp/dev/latest/usage/general.html#_CPPv4N8Trueface20ConfigurationOptions18smallestFaceHeightE
    Trueface::SDK tfSdk(options);

    // TODO: replace <LICENSE_CODE> with your license code.
    auto isValid = tfSdk.setLicense("<LICENSE_CODE>");
    if (!isValid) {
        std::cout << "Error: the provided license is invalid\n";
        return -1;
    }

    // Create a new database
    // This step is not required if using Trueface::DatabaseManagementSystem::NONE
    const std::string databaseName = "myDatabase.db";
    auto errorCode = tfSdk.createDatabaseConnection(databaseName);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: unable to create database\n";
        return -1;
    }

    // Create a collection
    const std::string collectioName = "myCollection";
    errorCode = tfSdk.createLoadCollection(collectioName);

    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: unable to create collection\n";
        return -1;
    }

    // Load the image / images we want to enroll
    errorCode = tfSdk.setImage("../../../images/obama/obama1.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: unable to read image\n";
        return -1;
    }

    // Detect the face in the image
    Trueface::FaceBoxAndLandmarks faceBoxAndLandmarks;
    bool faceDetected;
    errorCode = tfSdk.detectLargestFace(faceBoxAndLandmarks, faceDetected);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "There was an error with the call to detectLargestFace\n";
        return -1;
    }

    if (!faceDetected) {
        std::cout << "Unable to detect face\n";
        return -1;
    }

    // We want to only enroll high quality images into the database / collection
    // Therefore, ensure that the face height is at least 100px
    auto faceHeight = faceBoxAndLandmarks.bottomRight.y - faceBoxAndLandmarks.topLeft.y;
    std::cout << "Face height: " << faceHeight << std::endl;
    if (faceHeight < 100) {
        std::cout << "The face is too small in the image for a high quality enrollment." << std::endl;
        return -1;
    }

    // Get the aligned face chip so that we can compute the image quality
    uint8_t alignedImage[37632];
    errorCode = tfSdk.extractAlignedFace(faceBoxAndLandmarks, alignedImage);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "There was an error extracting the aligned face\n";
        return -1;
    }

    float quality;
    errorCode = tfSdk.estimateFaceImageQuality(alignedImage, quality);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute image quality\n";
        return -1;
    }

    // Ensure the image quality is above a threshold (TODO: adjust this threshold based on your use case).
    // Once again, we only want to enroll high quality images into our collection
    std::cout << "Face quality: " << quality << std::endl;
    if (quality < 0.8) {
        std::cout << "Please choose a higher quality enrollment image\n";
        return -1;
    }

    // As a final check, we can check the orientation of the head and ensure that it is facing forward
    // To see the effect of yaw and pitch on the match score, refer to: https://reference.trueface.ai/cpp/dev/latest/usage/face.html#_CPPv4N8Trueface3SDK23estimateHeadOrientationERK19FaceBoxAndLandmarksRfRfRf
    float yaw, pitch, roll;
    errorCode = tfSdk.estimateHeadOrientation(faceBoxAndLandmarks, yaw, pitch, roll);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute head orientation\n";
        return -1;
    }

    std::cout << "Yaw: " << yaw * 180 / M_PI  << ", Pitch: " << pitch * 180 / M_PI << ", Roll: " << roll * 180 / M_PI << " degrees" << std::endl;

    // TODO: Can filter out images with extreme yaw and pitch here

    // Generate the enrollment template
    Trueface::Faceprint enrollmentFaceprint;
    errorCode = tfSdk.getFaceFeatureVector(alignedImage, enrollmentFaceprint);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to generate template\n";
        return -1;
    }

    // Add the enrollment template to our collection
    // Any data that is added to the collection will persist after the application is terminated because of the DatabaseManagementSystem we chose.
    std::string UUID;
    errorCode = tfSdk.enrollFaceprint(enrollmentFaceprint, "Obama", UUID);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to enroll template\n";
        return -1;
    }
    // Can add other template pairs to the collection here...

    // TODO: Add the path to your video below
    cv::VideoCapture cap("../../../images/obama/speech.mp4");
    cv::VideoWriter outputVideo;

    cv::Size frameSize (static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    int ex = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

    // Name of our output file
    outputVideo.open("annotated.mp4" , ex, cap.get(cv::CAP_PROP_FPS), frameSize, true);

    while(true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Set the image using the capture frame buffer
        errorCode = tfSdk.setImage(frame.data, frame.cols, frame.rows, Trueface::ColorCode::bgr);
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "There was an error setting the image\n";
            return -1;
        }

        // Get all the bounding boxes
        std::vector<Trueface::FaceBoxAndLandmarks> bboxVec;
        tfSdk.detectFaces(bboxVec);

        // For each bounding box, get the face feature vector
        std::vector<Trueface::Faceprint> faceprints;
        faceprints.reserve(bboxVec.size());

        for (const auto &bbox: bboxVec) {
            // Get the face feature vector
            Trueface::Faceprint tmpFaceprint;
            tfSdk.getFaceFeatureVector(bbox, tmpFaceprint);
            faceprints.emplace_back(std::move(tmpFaceprint));
        }

        // Run batch identification on the faceprints
        std::vector<bool> found;
        std::vector<Trueface::Candidate> candidates;
        tfSdk.batchIdentifyTopCandidate(faceprints, candidates, found, threshold);

        // If the identity was found, draw the identity label
        // If the identity was not found, blur the face

        for (size_t i = 0; i < found.size(); ++i) {
            const auto& bbox = bboxVec[i];
            const auto& candidate = candidates[i];

            if (found[i]) {
                // If the similarity is greater than our threshold, then we have a match
                cv::Point topLeft(bbox.topLeft.x, bbox.topLeft.y);
                cv::Point bottomRight(bbox.bottomRight.x, bbox.bottomRight.y);
                cv::Scalar color (0, 255, 0);
                cv::rectangle(frame, topLeft, bottomRight, color, 2);
                setLabel(frame, candidate.identity, topLeft, color);
            } else {
                // If the identity is not found, draw a white bounding box
                cv::Point topLeft(bbox.topLeft.x, bbox.topLeft.y);
                cv::Point bottomRight(bbox.bottomRight.x, bbox.bottomRight.y);
                cv::Scalar color (255, 255, 255);
                cv::rectangle(frame, topLeft, bottomRight, color, 2);
            }
        }

        cv::imshow("frame", frame);
        outputVideo.write(frame);

        if (cv::waitKey(1) == 27) {
            break; // stop capturing by pressing ESC
        }

    }

    return 0;
}