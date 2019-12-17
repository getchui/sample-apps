#include <opencv2/opencv.hpp>
#include <vector>

#include "tf_sdk.h"
#include "tf_data_types.h"

int main() {
    Trueface::SDK tfSdk;

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense("<LICENSE_CODE>");
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

        // Display the landmark locations and bounding box on the image
        for (const auto& landmarks: landmarksVec) {
            // Only use the landmarks / bounding box if the score is above 0.90
            if (landmarks.score < 0.90)
                continue;

            // Draw a rectangle using the top left and bottom right coordinates of the bounding box
            cv::Point topLeft(landmarks.topLeft.x, landmarks.topLeft.y);
            cv::Point bottomRight(landmarks.bottomRight.x, landmarks.bottomRight.y);
            cv::rectangle(frame, topLeft, bottomRight, cv::Scalar(255, 0, 0), 2);

            // Draw the facial landmarks
            // the facial landmark points: left eye, right eye, nose, left mouth corner, right mouth corner
            for (const auto& landmark: landmarks.landmarks) {
                cv::Point p(landmark.x, landmark.y);
                cv::circle(frame, p, 1, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
            }
        }

        cv::imshow("frame", frame);

        if (cv::waitKey(10) == 27) {
            break; // stop capturing by pressing ESC
        }
    }

    return 0;
}