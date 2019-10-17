#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

#include "tf_sdk.h"
#include "tf_data_types.h"

int main() {
    Trueface::SDK tfSdk;

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense("<license>");
    if (!isValid) {
        std::cout << "Error: the provided license is invalid\n";
        return -1;
    }

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

        // Get the landmark locations
        std::vector<Trueface::FaceBoxAndLandmarks> landmarksVec;
        tfSdk.detectFaces(landmarksVec);

        // Use the landmark locations to obtain the yaw, pitch, and roll
        for (const auto& landmarks: landmarksVec) {
            // Only use the landmarks / bounding box if the score is above 0.90
            if (landmarks.score < 0.90)
                continue;

            // Compute the yaw, pitch, roll
            float yaw, pitch, roll;
            auto retCode = tfSdk.estimateHeadOrientation(landmarks, yaw, pitch, roll);
            if (retCode != Trueface::ErrorCode::NO_ERROR) {
                std::cout << "Unable to compute orientation\n";
                continue;
            }

            cv::Point nosePoint(landmarks.landmarks[2].x, landmarks.landmarks[2].y);
            cv::arrowedLine(frame, nosePoint, cv::Point(cos(roll) * cos(yaw) * 100 + nosePoint.x, sin(-roll) * 100 + nosePoint.y), cv::Scalar(255, 0, 0), 4);
            cv::arrowedLine(frame, nosePoint, cv::Point(cos(roll + M_PI / 2) * 100 + nosePoint.x, sin(-roll - M_PI/2) * sin(M_PI/2 - pitch) * 100 + nosePoint.y), cv::Scalar(0, 255, 0), 4);
            cv::arrowedLine(frame, nosePoint, cv::Point(cos(M_PI/2 - yaw) * 100 + nosePoint.x, sin(M_PI - pitch) * 100 + nosePoint.y), cv::Scalar(0, 0, 255), 4);

        }

        cv::imshow("frame", frame);

        if (cv::waitKey(10) == 27) {
            break; // stop capturing by pressing ESC
        }
    }

    return 0;
}