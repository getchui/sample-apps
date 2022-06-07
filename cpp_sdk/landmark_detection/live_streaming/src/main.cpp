#include <opencv2/opencv.hpp>
#include <vector>
#include <thread>
#include <mutex>
#include <memory>
#include <chrono>
#include <atomic>

#include "tf_sdk.h"
#include "tf_data_types.h"

using namespace Trueface;

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

    // Start by specifying the configuration options to be used.
    // Can choose to use default configuration options if preferred by calling the default SDK constructor.
    // Learn more about configuration options here: https://reference.trueface.ai/cpp/dev/latest/usage/general.html
    ConfigurationOptions options;
    // The face recognition model to use. Use the most accurate face recognition model.
    options.frModel = FacialRecognitionModel::TFV5;
    // The object detection model to use.
    options.objModel = ObjectDetectionModel::ACCURATE;
    // The face detection filter.
    options.fdFilter = FaceDetectionFilter::BALANCED;
    // Smallest face height in pixels for the face detector.
    options.smallestFaceHeight = 16;
    // The path specifying the directory where the model files have been downloaded
    options.modelsPath = "./";
    auto modelsPath = std::getenv("MODELS_PATH");
    if (modelsPath) {
        options.modelsPath = modelsPath;
    }
    // Enable vector compression to improve 1 to 1 comparison speed and 1 to N search speed.
    options.frVectorCompression = false;
    // Database management system for storage of biometric templates for 1 to N identification.
    options.dbms = DatabaseManagementSystem::SQLITE;

    // If encryption is enabled, must provide an encryption key
    options.encryptDatabase.enableEncryption = false;
    options.encryptDatabase.key = "TODO: Your encryption key here";

    // Initialize module in SDK constructor.
    // By default, the SDK uses lazy initialization, meaning modules are only initialized when they are first used (on first inference).
    // This is done so that modules which are not used do not load their models into memory, and hence do not utilize memory.
    // The downside to this is that the first inference will be much slower as the model file is being decrypted and loaded into memory.
    // Therefore, if you know you will use a module, choose to pre-initialize the module, which reads the model file into memory in the SDK constructor.
    InitializeModule initializeModule;
    initializeModule.faceDetector = true;
    options.initializeModule = initializeModule;

    // Options for enabling GPU
    // We will disable GPU inference, but you can easily enable it by modifying the following options
    // Note, you may require a specific GPU enabled token in order to enable GPU inference.
    GPUModuleOptions gpuOptions;
    gpuOptions.enableGPU = false; // TODO: Change this to true to enable GPU inference.
    gpuOptions.maxBatchSize = 4;
    gpuOptions.optBatchSize = 1;
    gpuOptions.maxWorkspaceSizeMb = 2000;
    gpuOptions.deviceIndex = 0;
    gpuOptions.precision = Precision::FP16;

    options.gpuOptions.faceRecognizerGPUOptions = gpuOptions;
    options.gpuOptions.faceDetectorGPUOptions = gpuOptions;

    // Alternatively, can also do the following to enable GPU inference for all supported modules:
//    options.gpuOptions = true;

    SDK tfSdk(options);

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense(TRUEFACE_TOKEN);
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
        TFImage img;
        auto errorCode = tfSdk.preprocessImage(frame.data, frame.cols, frame.rows, ColorCode::bgr, img);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "There was an error setting the image\n";
            return -1;
        }

        // Get the landmark locations
        std::vector<FaceBoxAndLandmarks> landmarksVec;
        tfSdk.detectFaces(img, landmarksVec);

        // Display the landmark locations and bounding box on the image
        for (const auto& faceBoxAndLandmarks: landmarksVec) {

            // Draw a rectangle using the top left and bottom right coordinates of the bounding box
            cv::Point topLeft(faceBoxAndLandmarks.topLeft.x, faceBoxAndLandmarks.topLeft.y);
            cv::Point bottomRight(faceBoxAndLandmarks.bottomRight.x, faceBoxAndLandmarks.bottomRight.y);
            cv::rectangle(frame, topLeft, bottomRight, cv::Scalar(255, 0, 0), 2);

            Landmarks landmarks;
            errorCode = tfSdk.getFaceLandmarks(img, faceBoxAndLandmarks, landmarks);
            // Draw the 106 face landmarks
            for (const auto& landmark: landmarks) {
                cv::Point p(landmark.x, landmark.y);
                cv::circle(frame, p, 2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
            }

            // Draw the 5 facial landmarks
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