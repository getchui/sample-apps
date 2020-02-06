#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "tf_sdk.h"
#include "tf_data_types.h"

// Utility function for drawing the label on our image
void setLabel(cv::Mat& im, const std::string label, const cv::Point & origin) {
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.6;
    const int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, font, scale, thickness, &baseline);
    cv::rectangle(im, origin + cv::Point(0, baseline), origin + cv::Point(text.width, -text.height), CV_RGB(0,0,0), cv::FILLED);
    cv::putText(im, label, origin, font, scale, CV_RGB(255,255,255), thickness, cv::LINE_AA);
}


int main() {
    // Threshold used to determine if it is a match
    const float threshold = 0.6;

    // Gallery used to store our templates
    std::vector<std::pair<Trueface::Faceprint, std::string>> gallery;

    Trueface::SDK tfSdk;

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense("<LICENSE_CODE>");
    if (!isValid) {
        std::cout << "Error: the provided license is invalid\n";
        return -1;
    }

    // Load the image / images we want to enroll
    auto errorCode = tfSdk.setImage("../../../images/armstrong/armstrong1.jpg");
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

    // Add the enrollment template to our gallery
    gallery.emplace_back(enrollmentFaceprint, "Armstrong");

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
        auto errorCode = tfSdk.setImage(frame.data, frame.cols, frame.rows, Trueface::ColorCode::bgr);
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
            uint8_t *alignedChip = new uint8_t[imgSize];
            tfSdk.extractAlignedFace(bbox, alignedChip);

            // Generate a template from the aligned chip
            Trueface::Faceprint faceprint;
            const auto err = tfSdk.getFaceFeatureVector(alignedChip, faceprint);
            delete[] alignedChip;

            if (err != Trueface::ErrorCode::NO_ERROR)
                continue;

            // Compare the template to those in our gallery
            float maxScore = 0;
            int maxIdx = 0;

            // Iterate through the gallery to find the template with the highest match probability
            // If the match probability is greater than our threshold, then we know it's a match
            for (int i = 0; i < gallery.size(); ++i) {
                float matchProbability;
                float similarityMeasure;

                auto returnCode = tfSdk.getSimilarity(gallery[i].first, faceprint, matchProbability, similarityMeasure);
                if (returnCode != Trueface::ErrorCode::NO_ERROR)
                    continue;

                if (matchProbability > maxScore) {
                    maxScore = matchProbability;
                    maxIdx = i;
                }
            }

            if (maxScore > threshold) {
                // We have a match
                // Display the bounding box and label
                cv::Point topLeft(bbox.topLeft.x, bbox.topLeft.y);
                cv::Point bottomRight(bbox.bottomRight.x, bbox.bottomRight.y);
                cv::rectangle(frame, topLeft, bottomRight, cv::Scalar(255, 0, 0));

                setLabel(frame, gallery[maxIdx].second, topLeft);
            } else {

                // If the face has not been enrolled in our database, blur the face
                cv::Rect blurRect(bbox.topLeft.x, bbox.topLeft.y, bbox.bottomRight.x - bbox.topLeft.x, bbox.bottomRight.y - bbox.topLeft.y);
                cv::GaussianBlur(frame(blurRect), frame(blurRect), cv::Size(0, 0), 50);
            }

        }

        cv::imshow("frame", frame);

        if (cv::waitKey(10) == 27) {
            break; // stop capturing by pressing ESC
        }

    }
    return 0;
}