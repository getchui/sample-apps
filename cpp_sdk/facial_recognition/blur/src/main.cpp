#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

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

    Trueface::ConfigurationOptions options;
    options.frModel = Trueface::FacialRecognitionModel::FULL;
    options.dbms = Trueface::DatabaseManagementSystem::NONE; // The data will not persist
    Trueface::SDK tfSdk(options);

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense("<LICENSE_CODE>");
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
    errorCode = tfSdk.createCollection(collectioName);

    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: unable to create collection\n";
        return -1;
    }

    // Load the image / images we want to enroll
    errorCode = tfSdk.setImage("../../../images/armstrong/armstrong1.jpg");
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: unable to read image\n";
        return -1;
    }

    // Generate the enrollment template
    Trueface::Faceprint enrollmentFaceprint;
    errorCode = tfSdk.getLargestFaceFeatureVector(enrollmentFaceprint);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to generate template\n";
        return -1;
    }

    // Add the enrollment template to our collection
    // Any data that is added to the collection will persist after the application is terminated
    std::string UUID;
    errorCode = tfSdk.enrollTemplate(enrollmentFaceprint, "Armstrong", UUID);
    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to enroll template\n";
        return -1;
    }

    // Can add other template pairs to the gallery here...

    cv::VideoCapture cap;
    // Open the default camera, use something different from 0 otherwise
    if (!cap.open(0)) {
        std::cout << "Unable to open video capture\n";
        return -1;
    }

    while(true) {
        // Read the frame from the VideoCapture source
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            break; // End of video stream
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
            // Obtain the aligned chip
            uint8_t alignedChip[112 * 112 * 3];
            tfSdk.extractAlignedFace(bbox, alignedChip);

            // Get the face feature vector
            Trueface::Faceprint tmpFaceprint;
            tfSdk.getFaceFeatureVector(alignedChip, tmpFaceprint);
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
                // The identity was found, draw the identity label
                cv::Point topLeft(bbox.topLeft.x, bbox.topLeft.y);
                cv::Point bottomRight(bbox.bottomRight.x, bbox.bottomRight.y);
                cv::Scalar color(0, 255, 0);
                cv::rectangle(frame, topLeft, bottomRight, color, 2);
                setLabel(frame, candidate.identity, topLeft, color);
            } else {
                // The identity was not found, blur the face
                // Can change the blur size depending on the level of blur desired
                size_t blurSize = 45;

                // The identity was not found, blur the face
                cv::Rect blurRect(bbox.topLeft.x, bbox.topLeft.y, bbox.bottomRight.x - bbox.topLeft.x, bbox.bottomRight.y - bbox.topLeft.y);
                cv::blur(frame(blurRect), frame(blurRect), cv::Size(blurSize, blurSize));

                // Draw white rectangle around face
                cv::Point topLeft(bbox.topLeft.x, bbox.topLeft.y);
                cv::Point bottomRight(bbox.bottomRight.x, bbox.bottomRight.y);
                cv::Scalar color(255, 255, 255);
                cv::rectangle(frame, topLeft, bottomRight, color, 2);
            }
        }

        cv::imshow("frame", frame);

        if (cv::waitKey(10) == 27) {
            break; // stop capturing by pressing ESC
        }

    }
    return 0;
}