#include <opencv2/opencv.hpp>
#include <vector>
#include <thread>
#include <mutex>
#include <memory>
#include <chrono>
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
        // Open the video capture
        // Open the default camera (TODO: Can change the camera source, for example to an RTSP stream)
        if (!m_cap.open(0)) {
            throw std::runtime_error("Unable to open video capture");
        }

        // Launch a thread to start grab the newest frame from the frame buffer
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
        double fps = m_cap.get(cv::CAP_PROP_FPS);
        int sleepDurationMs = static_cast<int>(1000 / fps);

        while(m_run) {
            // Sleep so that we match the camera frame rate
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepDurationMs));

            // Grab a frame from the frame buffer, discard the return value
            // This will discard built up frames in the buffer to ensure we only process the latest frame to remove any delay
            m_mtx.lock();
            m_cap.grab();
            m_mtx.unlock();
        }
    }

    // Returns true if a frame was grabber
    bool grabFrame(cv::Mat& frame) {
        const std::lock_guard<std::mutex> lock(m_mtx);
        return m_cap.retrieve(frame);
    }
private:
    std::unique_ptr<std::thread> m_rtspThread = nullptr;
    std::mutex m_mtx;
    std::atomic<bool>& m_run;
    cv::VideoCapture m_cap;
};


int main() {
    std::atomic<bool> run {true};
    StreamController streamController(run);

    Trueface::ConfigurationOptions options;
    options.fdFilter = Trueface::FaceDetectionFilter::BALANCED; // https://reference.trueface.ai/cpp/dev/latest/usage/general.html#_CPPv4N8Trueface19FaceDetectionFilterE
    options.fdMode = Trueface::FaceDetectionMode::VERSATILE; // https://reference.trueface.ai/cpp/dev/latest/usage/general.html#_CPPv4N8Trueface17FaceDetectionModeE

    Trueface::SDK tfSdk(options);

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense("<LICENSE_CODE>");
    if (!isValid) {
        std::cout << "Error: the provided license is invalid\n";
        return -1;
    }

    while(run) {
        // Grab the latest frame from the video stream
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
        for (const auto& faceBoxAndLandmarks: landmarksVec) {

            // Draw a rectangle using the top left and bottom right coordinates of the bounding box
            cv::Point topLeft(faceBoxAndLandmarks.topLeft.x, faceBoxAndLandmarks.topLeft.y);
            cv::Point bottomRight(faceBoxAndLandmarks.bottomRight.x, faceBoxAndLandmarks.bottomRight.y);
            cv::rectangle(frame, topLeft, bottomRight, cv::Scalar(255, 0, 0), 2);

            // Draw the facial landmarks
            // the facial landmark points: left eye, right eye, nose, left mouth corner, right mouth corner
            for (const auto& landmark: faceBoxAndLandmarks.landmarks) {
                cv::Point p(landmark.x, landmark.y);
                cv::circle(frame, p, 2, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
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