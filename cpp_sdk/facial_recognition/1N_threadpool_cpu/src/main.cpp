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

using namespace Trueface;

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
        // Start by specifying the configuration options to be used.
        // Can choose to use default configuration options if preferred by calling the default SDK constructor.
        // Learn more about configuration options here: https://reference.trueface.ai/cpp/dev/latest/usage/general.html
        ConfigurationOptions options;
        // The face recognition model to use. Use the most accurate face recognition model.
        options.frModel = FacialRecognitionModel::LITE_V2;
        // The object detection model to use.
        options.objModel = ObjectDetectionModel::ACCURATE;
        // The face detection filter.
        options.fdFilter = FaceDetectionFilter::BALANCED;
        // Smallest face height in pixels for the face detector.
        options.smallestFaceHeight = 40;
        // The path specifying the directory where the model files have been downloaded
        options.modelsPath = "./";
        auto modelsPath = std::getenv("MODELS_PATH");
        if (modelsPath) {
            options.modelsPath = modelsPath;
        }
        // Enable vector compression to improve 1 to 1 comparison speed and 1 to N search speed.
        options.frVectorCompression = false;
        // Database management system for storage of biometric templates for 1 to N identification.
        options.dbms = DatabaseManagementSystem::POSTGRESQL;

        // Encrypt the biometric templates stored in the database
        EncryptDatabase encryptDatabase;
        encryptDatabase.enableEncryption = false; // TODO: To encrypt the database change this to true
        encryptDatabase.key = "TODO: Your encryption key here";
        options.encryptDatabase = encryptDatabase;

        // Initialize module in SDK constructor.
        // By default, the SDK uses lazy initialization, meaning modules are only initialized when they are first used (on first inference).
        // This is done so that modules which are not used do not load their models into memory, and hence do not utilize memory.
        // The downside to this is that the first inference will be much slower as the model file is being decrypted and loaded into memory.
        // Therefore, if you know you will use a module, choose to pre-initialize the module, which reads the model file into memory in the SDK constructor.
        InitializeModule initializeModule;
        initializeModule.faceDetector = true;
        initializeModule.faceRecognizer = true;
        options.initializeModule = initializeModule;

        // Options for enabling GPU
        // We will disable GPU inference, but you can easily enable it by modifying the following options
        // Note, you may require a specific GPU enabled token in order to enable GPU inference.
        options.gpuOptions = false; // TODO: Change this to true to enable GPU inference
        options.gpuOptions.deviceIndex = 0;

        GPUModuleOptions moduleOptions;
        moduleOptions.maxBatchSize = 4;
        moduleOptions.optBatchSize = 1;
        moduleOptions.maxWorkspaceSizeMb = 2000;
        moduleOptions.precision = Precision::FP16;

        options.gpuOptions.faceRecognizerGPUOptions = moduleOptions;
        options.gpuOptions.faceDetectorGPUOptions = moduleOptions;
        options.gpuOptions.maskDetectorGPUOptions = moduleOptions;

        // Create the SDK instance
        m_sdkPtr = std::make_unique<SDK>(options);

        auto valid = m_sdkPtr->setLicense(sdkToken);
        if (!valid) {
            throw std::runtime_error("Token is not valid!");
        }

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
            std::thread t(&Controller::detectAndEnqueueFaces, this);
            m_workerThreads.emplace_back(std::move(t));
        }

        // Create template extraction worker threads
        size_t numTemplateExtractionWorkerThreads = 3;
        for (size_t i = 0; i < numTemplateExtractionWorkerThreads; ++i) {
            std::thread t(&Controller::extractAndEnqueueTemplate, this);
            m_workerThreads.emplace_back(std::move(t));
        }

        // Create identification worker thread
        size_t numIdentificationWorkerThreads = 1;
        for (size_t i = 0; i < numIdentificationWorkerThreads; ++i) {
            std::thread t (&Controller::identifyTemplate, this, databaseConnectionURL, collectionName, i == 0);
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

    // Function for connecting to an RTSP stream and enrolling preproessed frames into a queue.
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

            // Preprocess the frame
            TFImage img;
            auto errorcode = m_sdkPtr->preprocessImage(frame.data, frame.cols, frame.rows, ColorCode::bgr, img);
            if (errorcode != ErrorCode::NO_ERROR) {
                std::cout << "Thread " << std::this_thread::get_id() << ": There was an error preprocessing the frame" << std::endl;
                std::cout << errorcode << std::endl;
                continue;
            }

            // Push a frame to the queue, notify thread pool that there is work
            {
                std::lock_guard<std::mutex> lock(m_imageQueueMtx);
                m_imageQueue.push(img);
            }

            m_imageQueueCondVar.notify_one();
        }

        std::cout << "RTSP thread " << std::this_thread::get_id() << " shutting down..." << std::endl;
    }

    // Function for searching for all faces in the frame and pushing the aligned face chips
    // into another queue to be processed for face recognition
    void detectAndEnqueueFaces() {
        // Main loop
        while(m_run) {
            TFImage img;
            // Wait for work
            {
                std::unique_lock<std::mutex> lock(m_imageQueueMtx);
                m_imageQueueCondVar.wait(lock, [this]{ return !this->m_imageQueue.empty() || !this->m_run;});

                if (!m_run) {
                    // Exit signal received
                    break;
                }

                img = m_imageQueue.front();
                m_imageQueue.pop();
            }

            // Pass the image to the SDK, run face detection
            std::vector<FaceBoxAndLandmarks> faceBoxAndLandmarks;
            auto retcode = m_sdkPtr->detectFaces(img, faceBoxAndLandmarks);

            if (retcode != ErrorCode::NO_ERROR) {
                std::cout << "Thread " << std::this_thread::get_id() << ": Error detecting faces" << std::endl;
                continue;
            }

            // For each detected face, extract the aligned face chip, add to the face chip queue
            // Each face chip is 112x112 pixels in size, so we must allocate 112x112x3 bytes
            for (const auto& fb: faceBoxAndLandmarks) {
                TFFacechip facechip;
                retcode = m_sdkPtr->extractAlignedFace(img, fb, facechip);

                if (retcode != ErrorCode::NO_ERROR) {
                    std::cout << "Thread " << std::this_thread::get_id() << ": Error extracting aligned face" << std::endl;
                    continue;
                }

                // Push the face image into our queue and indicate that work is ready
                {
                    std::lock_guard<std::mutex> lock(m_faceChipQueueMtx);
                    m_faceChipQueue.push(facechip);
                }
                m_faceChipQueueCondVar.notify_one();
            }
        }
        std::cout << "Face detection thread " << std::this_thread::get_id() << " shutting down..." << std::endl;
    }

    // Function for generating a face recognition template from the face chips.
    // The face templates are then added to a queue to be processed for identification
    void extractAndEnqueueTemplate() {
        // Main loop
        while (m_run) {
            TFFacechip facechip;
            // wait for work
            {
                std::unique_lock<std::mutex> lock(m_faceChipQueueMtx);
                m_faceChipQueueCondVar.wait(lock, [this] {return !this->m_faceChipQueue.empty() || !this->m_run;});

                if (!m_run) {
                    // Exit signal received
                    break;
                }

                facechip = std::move(m_faceChipQueue.front());
                m_faceChipQueue.pop();
            }

            // Generate a face recognition template from the face image
            Faceprint faceprint;
            auto retcode = m_sdkPtr->getFaceFeatureVector(facechip, faceprint);
            if (retcode != ErrorCode::NO_ERROR) {
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

    void identifyTemplate(const std::string& databaseURL, const std::string& collectionName, bool first) {
        // As long as all instances of the SDK are in the same process, then only one instance needs to connect to the database
        // To learn more, read the top of: https://reference.trueface.ai/cpp/dev/latest/usage/identification.html
        if (first) {
            auto retcode = m_sdkPtr->createDatabaseConnection(databaseURL);
            if (retcode != ErrorCode::NO_ERROR) {
                throw std::runtime_error("Unable to connect to database");
            }

            retcode = m_sdkPtr->createLoadCollection(collectionName);
            if (retcode != ErrorCode::NO_ERROR) {
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
            Faceprint faceprint;
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
            Candidate candidate;
            bool found;
            auto retcode = m_sdkPtr->identifyTopCandidate(faceprint, candidate, found);
            if (retcode != ErrorCode::NO_ERROR) {
                std::cout << "Unable to run identify top candidate" << std::endl;
                continue;
            }

            if (found) {
                // A match was found
                // TODO: Do something with match information, run callback function, etc
                // For the sake of the demo, we will just log it to the console
                std::cout << "Match found: " << candidate.identity << " with " << candidate.matchProbability * 100 << "% probability" << std::endl;
            }
        }
        std::cout << "Identify thread " << std::this_thread::get_id() << " shutting down..." << std::endl;
    }

    // Single SDK instance to be used by various threads
    std::unique_ptr<SDK> m_sdkPtr = nullptr;

    // Shared queues and their mutexes
    std::queue<TFImage> m_imageQueue;
    std::queue<TFFacechip> m_faceChipQueue;
    std::queue<Faceprint> m_faceprintQueue;

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
    const std::string token = TRUEFACE_TOKEN;

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
