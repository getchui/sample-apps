#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

#include "tf_sdk.h"
#include "tf_data_types.h"

// Utility class used for computing the running average
class RunningAvg {
public:
    explicit RunningAvg(size_t maxElem);
    float addVal(float val);
private:
    const size_t m_maxElem;
    std::vector<float> m_data;
    size_t m_pos = 0;
};

int main() {
    // Set the number of frames we want to average for yaw, pitch, and roll to remove noise
    const size_t NUM_ELEM = 10;
    RunningAvg yawAvg(NUM_ELEM);
    RunningAvg pitchAvg(NUM_ELEM);
    RunningAvg rollAvg(NUM_ELEM);

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

        // Get the landmarks for the largest face
        bool found;
        Trueface::FaceBoxAndLandmarks landmarks;
        tfSdk.detectLargestFace(landmarks, found);

        // Use the landmark locations to obtain the yaw, pitch, and roll
        if (found) {

            // Compute the yaw, pitch, roll
            float yaw, pitch, roll;
            auto retCode = tfSdk.estimateHeadOrientation(landmarks, yaw, pitch, roll);
            if (retCode != Trueface::ErrorCode::NO_ERROR) {
                std::cout << "Unable to compute orientation\n";
                continue;
            }

            // Get the running average of the values
            yaw = yawAvg.addVal(yaw);
            pitch = pitchAvg.addVal(pitch);
            roll = rollAvg.addVal(roll);

            // Center point for the axis we will draw
            const cv::Point origin(100, 100);

            // Compute 3D rotation axis from yaw, pitch, roll
            // https://stackoverflow.com/a/32133715/4943329
            const auto x1 = 100 * cos(yaw) * cos(roll);
            const auto y1 = 100 * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw));

            const auto x2 = 100 * (-1 * cos(yaw) * sin(roll));
            const auto y2 = 100 * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll));

            const auto x3 = 100 * sin(yaw);
            const auto y3 = 100 * (-1 * cos(yaw) * sin(pitch));

            // Draw the arrows on the screen
            cv::arrowedLine(frame, origin, cv::Point(x1 + origin.x, y1 + origin.y), cv::Scalar(255, 0, 0), 4);
            cv::arrowedLine(frame, origin, cv::Point(x2 + origin.x, y2 + origin.y), cv::Scalar(0, 255, 0), 4);
            cv::arrowedLine(frame, origin, cv::Point(x3 + origin.x, y3 + origin.y), cv::Scalar(0, 0, 255), 4);
        }

        cv::imshow("frame", frame);

        if (cv::waitKey(10) == 27) {
            break; // stop capturing by pressing ESC
        }
    }

    return 0;
}

RunningAvg::RunningAvg(size_t maxElem)
        : m_maxElem (maxElem)
{}

// Use circular buffer to compute the average
float RunningAvg::addVal(float val) {
    if (m_data.size() <= m_maxElem) {
        m_data.emplace_back(val);
    } else {
        m_data[m_pos++] = val;
    }

    if (m_pos == m_maxElem)
        m_pos = 0;

    float sum = 0;
    for (const auto& x: m_data) {
        sum += x;
    }

    return sum / m_data.size();
}