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
#include <csignal>
#include <unistd.h>

#include "tf_sdk.h"
#include "tf_data_types.h"

bool g_run = true;
std::mutex g_mtx;
std::condition_variable g_conditionVariable;

void sigstop(int a) {
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_run = false;
    }
    g_conditionVariable.notify_one();
}

class Controller {
public:
    Controller(const std::string& sdkToken, const std::vector<std::string>& rtspURLs,
               const std::string& databaseConnectionURL, const std::string& collectionName) {
        // Choose our SDK configuration options
        Trueface::ConfigurationOptions options;
        // Since we are running on CPU only, use the lite model
        options.frModel = Trueface::FacialRecognitionModel::LITE; // If you want better accuracy, use the TFV5 model.
        // Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file
        options.smallestFaceHeight = 40;
        options.dbms = Trueface::DatabaseManagementSystem::POSTGRESQL;

        // Create our logging thread
        m_workerThreads.emplace_back(std::thread(&Controller::logQueueSizes, this));

        // Create a rtsp worker thread for each rtsp stream
        for (const auto& rtspURL: rtspURLs) {
            std::thread t(&Controller::grabAndEnqueueFrames, this, rtspURL);
            m_workerThreads.emplace_back(std::move(t));
        }

        // Create our face detection worker threads
        size_t numFaceDetectionWorkerThreads = 2;
        for (size_t i = 0; i < numFaceDetectionWorkerThreads; ++i) {
            std::thread t(&Controller::detectAndEnqueueFaces, this, sdkToken, options);
            m_workerThreads.emplace_back(std::move(t));
        }

        // Create template extraction worker threads
        size_t numTemplateExtractionWorkerThreads = 3;
        for (size_t i = 0; i < numTemplateExtractionWorkerThreads; ++i) {
            std::thread t(&Controller::extractAndEnqueueTemplate, this, sdkToken, options);
            m_workerThreads.emplace_back(std::move(t));
        }

        // Create identification worker thread
        size_t numIdentificationWorkerThreads = 1;
        for (size_t i = 0; i < numIdentificationWorkerThreads; ++i) {
            std::thread t (&Controller::identifyTemplate, this, sdkToken, options, databaseConnectionURL, collectionName, i == 0);
            m_workerThreads.emplace_back(std::move(t));
        }
    }

    ~Controller() {
        // Must check value of m_terminated before calling terminate()
        // because user of the API could have manually called terminate()
        if (!m_terminated) {
            terminate();
        }
    }

    // Signal to all the work threads that it's time to stop
    void terminate() {
        std::cout << "Terminate command received, shutting down all worker threads..." << std::endl;
        m_run = false;

        m_imageQueueCondVar.notify_all();
        m_faceChipQueueCondVar.notify_all();
        m_faceprintQueueCondVar.notify_all();

        // Wait for all of our threads
        for (auto& t: m_workerThreads) {
            if (t.joinable()) {
                t.join();
            }
        }

        m_workerThreads.clear();

        m_terminated = true;
    }
private:
    // Function for logging the queue sizes
    void logQueueSizes() {
        while(m_run) {
            // If the queue sizes get too large,
            // then you need to create more workers or reduce the number of input streams
            sleep(2);
            {
                std::lock_guard<std::mutex> lock (m_imageQueueMtx);
                std::cout << "Image Queue Size: " << m_imageQueue.size() << std::endl;
            }
            {
                std::lock_guard<std::mutex> lock (m_faceChipQueueMtx);
                std::cout << "Face Chip Queue Size: " << m_faceChipQueue.size() << std::endl;
            }
            {
                std::lock_guard<std::mutex> lock (m_faceprintQueueMtx);
                std::cout << "Faceprint Queue Size: " << m_faceprintQueue.size() << std::endl;
            }
        }
    }

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
            // Only retrieve ever 6th frame from the stream (5FPS)
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

        std::cout << "RTSP thread " << std::this_thread::get_id() << " shutting down..." << std::endl;
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
        std::cout << "Face detection thread " << std::this_thread::get_id() << " shutting down..." << std::endl;
    }

    // Function for generating a face recognition template from the face chips.
    // The face templates are then added to a queue to be processed for identification
    void extractAndEnqueueTemplate(const std::string& sdkToken, const Trueface::ConfigurationOptions& options) {
        // Create and initialize a Trueface SDK
        Trueface::SDK tfSdk(options);
        auto ret = tfSdk.setLicense(sdkToken);
        if (!ret) {
            throw std::runtime_error("Invalid token");
        }

        // Main loop
        while (m_run) {
            std::vector<uint8_t> faceImage;
            // wait for work
            {
                std::unique_lock<std::mutex> lock(m_faceChipQueueMtx);
                m_faceChipQueueCondVar.wait(lock, [this] {return !this->m_faceChipQueue.empty() || !this->m_run;});

                if (!m_run) {
                    // Exit signal received
                    break;
                }

                faceImage = std::move(m_faceChipQueue.front());
                m_faceChipQueue.pop();
            }

            // Generate a face recognition template from the face image
            Trueface::Faceprint faceprint;
            auto retcode = tfSdk.getFaceFeatureVector(faceImage.data(), faceprint);
            if (retcode != Trueface::ErrorCode::NO_ERROR) {
                std::cout << "Thread " << std::this_thread::get_id() << ": Unable to generate feature vector" << std::endl;
                continue;
            }

            // Push the faceprint into the queue and indicate that work is ready
            {
                std::lock_guard<std::mutex> lock (m_faceprintQueueMtx);
                m_faceprintQueue.push(std::move(faceprint));
            }

            m_faceprintQueueCondVar.notify_one();
        }
        std::cout << "Template extraction thread " << std::this_thread::get_id() << " shutting down..." << std::endl;
    }

    void identifyTemplate(const std::string& sdkToken, const Trueface::ConfigurationOptions& options,
                          const std::string& databaseURL, const std::string& collectionName, bool first) {
        // Create and initialize a Trueface SDK
        Trueface::SDK tfSdk(options);
        auto ret = tfSdk.setLicense(sdkToken);
        if (!ret) {
            throw std::runtime_error("Invalid token");
        }

        // As long as all instances of the SDK are in the same process, then only one instance needs to connect to the database
        // To learn more, read the top of: https://reference.trueface.ai/cpp/dev/latest/usage/identification.html
        if (first) {
            auto retcode = tfSdk.createDatabaseConnection(databaseURL);
            if (retcode != Trueface::ErrorCode::NO_ERROR) {
                throw std::runtime_error("Unable to connect to database");
            }

            retcode = tfSdk.createLoadCollection(collectionName);
            if (retcode != Trueface::ErrorCode::NO_ERROR) {
                throw std::runtime_error("Unable to create new collection or load existing collection");
            }

            {
                std::lock_guard<std::mutex> lock(m_databaseConnectionMtx);
                m_databaseConnected = true;
            }
            m_databaseConnectionConVar.notify_all();
        } else {
            // Wait for the first thread to connect to the database
            {
                std::unique_lock<std::mutex> lock (m_databaseConnectionMtx);
                m_databaseConnectionConVar.wait(lock, [this]{return this->m_databaseConnected;});
            }
        }


        // Main loop
        while(m_run) {
            Trueface::Faceprint faceprint;
            // Wait for work
            {
                std::unique_lock<std::mutex> lock(m_faceprintQueueMtx);
                m_imageQueueCondVar.wait(lock, [this]{ return !this->m_faceprintQueue.empty() || !this->m_run;});

                if (!m_run) {
                    // Exit signal received
                    break;
                }

                faceprint = std::move(m_faceprintQueue.front());
                m_faceprintQueue.pop();
            }

            // Run 1 to N identification
            Trueface::Candidate candidate;
            bool found;
            auto retcode = tfSdk.identifyTopCandidate(faceprint, candidate, found);
            if (retcode != Trueface::ErrorCode::NO_ERROR) {
                std::cout << "Unable to run identify top candidate" << std::endl;
            }

            if (found) {
                // A match was found
                // TODO: Do something with match information, run callback function, etc
                // For the sake of the demo, we will just log it to the console
                std::cout << "Match found: " << candidate.identity << " with " << candidate.matchProbability << " probability" << std::endl;
            }
        }
        std::cout << "Identify thread " << std::this_thread::get_id() << " shutting down..." << std::endl;
    }

    // Shared queues and their mutexes
    std::queue<cv::Mat> m_imageQueue;
    std::queue<std::vector<uint8_t>> m_faceChipQueue;
    std::queue<Trueface::Faceprint> m_faceprintQueue;

    std::mutex m_imageQueueMtx;
    std::mutex m_faceChipQueueMtx;
    std::mutex m_faceprintQueueMtx;
    std::mutex m_databaseConnectionMtx;

    // Conditional variables for indicating work is ready
    std::condition_variable m_imageQueueCondVar;
    std::condition_variable m_faceChipQueueCondVar;
    std::condition_variable m_faceprintQueueCondVar;
    std::condition_variable m_databaseConnectionConVar;

    // When set to false, worker threads should stop running
    std::atomic<bool> m_run {true};
    std::atomic<bool> m_terminated {false};
    bool m_databaseConnected  = false;

    // Worker threads
    std::vector<std::thread> m_workerThreads;
};

int main() {
    // The main thread will sleep until a stop signal is received
    signal(SIGINT, sigstop);
    signal(SIGQUIT, sigstop);
    signal(SIGTERM, sigstop);

    // TODO: Replace with your license token
    const std::string token = "<LICENSE_CODE>";

    // TODO: Fill out your database connection string and collection name
    // Should point to an existing collection that you have already filled with enrollment templates.
    const std::string postgresConnectionString = "host=localhost port=5432 dbname=my_database user=postgres password=admin";
    const std::string collectionName = "my_collection";

    // Simulate connecting to 5 different 1920x1080 resolution RTSP streams
    std::vector<std::string> rtspURLS = {
            "rtsp://root:!admin@192.168.0.11/stream1",
            "rtsp://root:!admin@192.168.0.11/stream1",
            "rtsp://root:!admin@192.168.0.11/stream1",
            "rtsp://root:!admin@192.168.0.11/stream1",
            "rtsp://root:!admin@192.168.0.11/stream1",
    };

    // Starts our main process
    Controller controller(token, rtspURLS, postgresConnectionString, collectionName);

    // Have the main thread sleep until kill signal received
    std::unique_lock<std::mutex> lock(g_mtx);
    g_conditionVariable.wait(lock, []{return !g_run;});
}
