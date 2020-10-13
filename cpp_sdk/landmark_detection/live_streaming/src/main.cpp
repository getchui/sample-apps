#include <opencv2/opencv.hpp>
#include <vector>
#include <thread>
#include <mutex>
#include <memory>
#include <atomic>

#include "tf_sdk.h"
#include "tf_data_types.h"

// Utility class for grabbing RTSP stream frames
// Runs in alternate thread to ensure we always have the latest frame from our RTSP stream
class StreamController{
public:
    explicit StreamController(std::atomic<bool>& run)
            : m_run(run)
    {
        // Launch a thread to start the rtsp processing
        m_rtspThread = std::make_unique<std::thread>(&StreamController::rtspThreadFunc, this);
    }

    ~StreamController() {
        // Stop the thread function loop
        m_run = false;
        if (m_rtspThread && m_rtspThread->joinable()) {
            m_rtspThread->join();
        }
    }

    void rtspThreadFunc() {
        cv::VideoCapture cap;
        // Open the default camera, use something different from 0 otherwise
        if (!cap.open(0)) {
            throw std::runtime_error("Unable to open video capture");
        }

        while(m_run) {
            // Read the frame from the VideoCapture source
            cv::Mat tempFrame;
            cap >> tempFrame;
            if (tempFrame.empty()) {
                m_run = false;
                break; // End of video stream
            }

            m_mtx.lock();
            m_frame = tempFrame;
            m_processed = false;
            m_mtx.unlock();
        }
    }

    // Returns true is the frame has not yet been processed
    // That way we can avoid running inference on the same frame multiple times (if our main loop is faster than the rtsp loop)
    bool grabFrame(cv::Mat& frame) {
        const std::lock_guard<std::mutex> lock(m_mtx);
        frame = m_frame;
        const auto hasProcessed = !m_processed;
        m_processed = true;
        return hasProcessed;
    }
private:
    std::unique_ptr<std::thread> m_rtspThread = nullptr;
    std::mutex m_mtx;
    cv::Mat m_frame;
    bool m_processed = true;
    std::atomic<bool>& m_run;
};


int main() {
    std::atomic<bool> run {true};
    StreamController streamController(run);

    Trueface::SDK tfSdk;

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense("<LICENSE_CODE>");
    if (!isValid) {
        std::cout << "Error: the provided license is invalid\n";
        return -1;
    }

    while(run) {
        // Grab the latest frame
        // Only process and display if it is a 'fresh' frame and has not already been processed
        cv::Mat frame;
        auto shouldProcess = streamController.grabFrame(frame);
        if (!shouldProcess) {
            continue;
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

        if (cv::waitKey(1) == 27) {
            run = false;
            break; // stop capturing by pressing ESC
        }
    }

    return 0;
}