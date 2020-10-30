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
        if (!m_cap.open("rtsp://root:!admin@192.168.0.11/stream1")) {
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
    options.smallestFaceHeight = 40; // https://reference.trueface.ai/cpp/dev/latest/usage/general.html#_CPPv4N8Trueface20ConfigurationOptions18smallestFaceHeightE

    Trueface::SDK tfSdk(options);

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbW90aW9uIjp0cnVlLCJmciI6dHJ1ZSwiYnlwYXNzX2dwdV91dWlkIjp0cnVlLCJwYWNrYWdlX2lkIjpudWxsLCJleHBpcnlfZGF0ZSI6IjIwMjEtMDEtMDEiLCJncHVfdXVpZCI6W10sInRocmVhdF9kZXRlY3Rpb24iOnRydWUsIm1hY2hpbmVzIjoxLCJhbHByIjp0cnVlLCJuYW1lIjoiQ3lydXMgR1BVIiwidGtleSI6Im5ldyIsImV4cGlyeV90aW1lX3N0YW1wIjoxNjA5NDU5MjAwLjAsImF0dHJpYnV0ZXMiOnRydWUsInR5cGUiOiJvZmZsaW5lIiwiZW1haWwiOiJjeXJ1c0B0cnVlZmFjZS5haSJ9.2uQZTG_AXcHVXEFahbvkM8-gmosLPxjSSnbfEAz5gpY");
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
        Trueface::FaceBoxAndLandmarks faceBoxAndLandmarks;
        bool found;
        tfSdk.detectLargestFace(faceBoxAndLandmarks, found);
        if (!found) {
            continue;
        }

        std::vector<Trueface::Point<int>> points;
        tfSdk.getFaceLandmarks(faceBoxAndLandmarks, points);


        // Display the landmark locations and bounding box on the image
        for (const auto& point: points) {
            cv::Point p(point.x, point.y);
            cv::circle(frame, p, 2, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        }

        cv::imshow("frame", frame);

        if (cv::waitKey(1) == 27) {
            run = false;
            break; // stop capturing by pressing ESC
        }
    }

    return 0;
}