#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "tf_sdk.h"
#include "tf_data_types.h"

// Utility function for drawing the label on our image
void setLabel(cv::Mat& im, const std::string& label, const cv::Point & origin) {
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.6;
    const int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, font, scale, thickness, &baseline);
    cv::rectangle(im, origin + cv::Point(0, baseline), origin + cv::Point(text.width, -text.height), CV_RGB(0,0,0), cv::FILLED);
    cv::putText(im, label, origin, font, scale, CV_RGB(255,255,255), thickness, cv::LINE_AA);
}


int main() {
    // TODO: Select a threshold for your application using the ROC curves
    // https://performance.trueface.ai/
    const float threshold = 0.6;

    Trueface::SDK tfSdk;

    // Create a collection
    const std::string collectionpath = "collection.db";
    auto errorCode = tfSdk.createCollection(collectionpath);

    if (errorCode != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Error: unable to create collection\n";
        return -1;
    }

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense("<LICENSE_CODE>");
    if (!isValid) {
        std::cout << "Error: the provided license is invalid\n";
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

        // For each bounding box, get the aligned chip
        for (auto &bbox: bboxVec) {

            const size_t imgSize = 112 * 112 * 3;
            auto *alignedChip = new uint8_t[imgSize];
            tfSdk.extractAlignedFace(bbox, alignedChip);

            // Generate a template from the aligned chip
            Trueface::Faceprint faceprint;
            const auto err = tfSdk.getFaceFeatureVector(alignedChip, faceprint);
            delete[] alignedChip;

            if (err != Trueface::ErrorCode::NO_ERROR)
                continue;

            // Run the identify function
            std::vector<Trueface::Candidate> candidates;
            errorCode = tfSdk.identify(faceprint, candidates);

            if (errorCode != Trueface::ErrorCode::NO_ERROR) {
                continue;
            }

            // If the similarity is greater than our threshold, then we have a match
            if (candidates[0].similarityMeasure > threshold) {
                cv::Point topLeft(bbox.topLeft.x, bbox.topLeft.y);
                cv::Point bottomRight(bbox.bottomRight.x, bbox.bottomRight.y);
                cv::rectangle(frame, topLeft, bottomRight, cv::Scalar(255, 0, 0), 2);
                setLabel(frame, candidates[0].identity, topLeft);
            } else {
                // If the face has not been enrolled in our database, blur the face
                cv::Rect blurRect(bbox.topLeft.x, bbox.topLeft.y, bbox.bottomRight.x - bbox.topLeft.x, bbox.bottomRight.y - bbox.topLeft.y);
                cv::blur(frame(blurRect), frame(blurRect), cv::Size(18, 18));
            }
        }

        cv::imshow("frame", frame);

        if (cv::waitKey(10) == 27) {
            break; // stop capturing by pressing ESC
        }

    }
    return 0;
}