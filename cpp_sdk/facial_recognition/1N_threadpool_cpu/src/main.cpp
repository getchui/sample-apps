#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>

#include "tf_sdk.h"
#include "tf_data_types.h"

class Controller {
public:
    Controller(const std::string& sdkToken, const std::vector<std::string>& rtspURLs) {
        // Choose our SDK configuration options
        Trueface::ConfigurationOptions options;
        options.frModel = Trueface::FacialRecognitionModel::FULL;
        options.smallestFaceHeight = 40;
        options.dbms = Trueface::DatabaseManagementSystem::POSTGRESQL;

        // Create our worker threads
        for (const auto& rtspURL: rtspURLs) {
            std::thread t(&Controller::grabAndEnqueueFrames, this, rtspURL);
            m_rtspWorkerThreads.emplace_back(std::move(t));
        }
    }

    ~Controller() {
        if (!m_terminated) {
            terminate();
        }
    }

    // Signal to all the work threads that it's time to stop
    void terminate() {
        m_run = false;

        m_imageQueueCondVar.notify_all();
        m_faceChipQueueCondVar.notify_all();

        // Wait for all of our threads
        for (auto& t: m_rtspWorkerThreads) {
            t.join();
        }

        m_rtspWorkerThreads.clear();

        m_terminated = true;
    }
private:
    // Function for connecting to an RTSP stream and enrolling frames into a queue.
    // Assuming our cameras stream at 30FPS, we will only process every 6th frame
    // to process at 5FPS because any higher and we end up processing very similar frames
    // and doing unnecessary work.
    void grabAndEnqueueFrames(const std::string& rtspURL) {
        // Open the video capture
        cv::VideoCapture cap;
        if (!cap.open(rtspURL)) {
            auto errMsg = "Unable to open video stream at URL: " + rtspURL;
            throw std::runtime_error(errMsg);
        }

        // Main loop
        while(m_run) {
            // Only retrieve ever 6th frame from the stream
            for (auto i = 0; i < 6; ++i) {
                cap.grab();
            }

            cv::Mat frame;
            auto ret = cap.retrieve(frame);
            if (!ret) {
                // Unable to retrieve frame
                continue;
            }

            // Push a frame to the queue, notify thread pool that there is work
            {
                std::lock_guard<std::mutex> lock(m_imageQueueMtx);
                m_imageQueue.push(std::move(frame));
            }

            m_imageQueueCondVar.notify_one();
        }

        std::cout << "Kill command received, rtsp thread shutting down..." << std::endl;
    }

    // Function for searching for all faces in the frame and pushing the aligned face chips
    // into another queue to be processed for face recognition
    void detectAndEnqueueFaces(const std::string& sdkToken, const Trueface::ConfigurationOptions& options) {
        // Create and initialize a Trueface SDK
        Trueface::SDK tfSdk(options);
        auto ret = tfSdk.setLicense(sdkToken);
        if (!ret) {
            throw std::runtime_error("Invalid token");
        }

        // Main loop
        while(m_run) {
            cv::Mat image;
            // Wait for work
            {
                std::unique_lock<std::mutex> lock(m_imageQueueMtx);
                m_imageQueueCondVar.wait(lock, [this]{ return !this->m_imageQueue.empty() || !this->m_run;});

                if (!m_run) {
                    // Exit signal received
                    break;
                }

                image = m_imageQueue.front();
                m_imageQueue.pop();
            }

            // Pass the image to the SDK, run face detection
            // OpenCV uses BGR default
            auto retcode = tfSdk.setImage(image.data, image.cols, image.rows, Trueface::ColorCode::bgr);
            if (retcode != Trueface::ErrorCode::NO_ERROR) {
                std::cout << "Thread " << std::this_thread::get_id() << ": Unable to set image" << std::endl;
                continue;
            }

            // Detect all the faces in the image
            std::vector<Trueface::FaceBoxAndLandmarks> faceBoxAndLandmarks;
            retcode = tfSdk.detectFaces(faceBoxAndLandmarks);

            if (retcode != Trueface::ErrorCode::NO_ERROR) {
                std::cout << "Thread " << std::this_thread::get_id() << ": Error detecting faces" << std::endl;
                continue;
            }

            // For each detected face, extract the aligned face chip, add to the face chip queue
            // Each face chip is 112x112 pixels in size, so we must allocate 112x112x3 bytes
            for (const auto& fb: faceBoxAndLandmarks) {
                std::vector<uint8_t> faceImage (112*112*3);
                retcode = tfSdk.extractAlignedFace(fb, faceImage.data());

                if (retcode != Trueface::ErrorCode::NO_ERROR) {
                    std::cout << "Thread " << std::this_thread::get_id() << ": Error extracting aligned face" << std::endl;
                    continue;
                }

                // Push the face image into our queue and indicate that work is ready
                {
                    std::lock_guard<std::mutex> lock(m_faceChipQueueMtx);
                    m_faceChipQueue.push(std::move(faceImage));
                }
                m_faceChipQueueCondVar.notify_one();
            }
        }


    }

    // Shared queues and their mutexes
    std::queue<cv::Mat> m_imageQueue;
    std::queue<std::vector<uint8_t>> m_faceChipQueue;

    std::mutex m_imageQueueMtx;
    std::mutex m_faceChipQueueMtx;

    // Conditional variables for indicating work is ready
    std::condition_variable m_imageQueueCondVar;
    std::condition_variable m_faceChipQueueCondVar;

    // When set to false, worker threads should stop running
    std::atomic<bool> m_run {true};
    bool m_terminated = false;

    // Worker threads
    std::vector<std::thread> m_rtspWorkerThreads;
};

int main() {
    // TODO Cyrus, remove token
    const std::string token = "";
    std::vector<std::string> rtspURLS = {
            "rtsp://root:!admin@192.168.0.11/stream1",
    };

    // Starts our main process
    Controller controller(token, rtspURLS);
}