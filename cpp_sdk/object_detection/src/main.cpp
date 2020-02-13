#include <opencv2/opencv.hpp>
#include <vector>

#include "tf_sdk.h"
#include "tf_data_types.h"

// Utility function for drawing the label on our image
    void setLabel(cv::Mat& im, const std::string& label, const cv::Point & origin) {
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.4;
    const int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, font, scale, thickness, &baseline);
    cv::rectangle(im, origin + cv::Point(0, baseline), origin + cv::Point(text.width, -text.height), CV_RGB(0,0,0), cv::FILLED);
    cv::putText(im, label, origin, font, scale, CV_RGB(255,255,255), thickness, cv::LINE_AA);
}


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

        // Detect the objects in the frame
        std::vector<Trueface::BoundingBox> bboxVec;
        tfSdk.detectObjects(bboxVec);

        // Display the bounding boxes and labels for the detected objects
        for (const auto& bbox: bboxVec) {
            // TODO: Can use the bbox.probability to filter results if desired

            // Draw a rectangle using the top left and bottom right coordinates of the bounding box
            cv::Point topLeft(bbox.topLeft.x, bbox.topLeft.y);
            cv::Point bottomRight(bbox.topLeft.x + bbox.width, bbox.topLeft.y + bbox.height);
            cv::rectangle(frame, topLeft, bottomRight, cv::Scalar(255, 0, 0), 2);

            // Convert the object label to a string
            // Draw the string on the frame
            const auto label = tfSdk.getObjectLabelString(bbox.label);
            setLabel(frame, label, topLeft);
        }

        cv::imshow("frame", frame);

        if (cv::waitKey(10) == 27) {
            break; // stop capturing by pressing ESC
        }
    }

    return 0;
}