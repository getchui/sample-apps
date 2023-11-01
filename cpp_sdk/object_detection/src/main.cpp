#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

#include "tf_data_types.h"
#include "tf_sdk.h"

using namespace Trueface;

// Utility class for grabbing RTSP stream frames
// Runs in alternate thread to ensure we always have the latest frame from our RTSP stream
class StreamController {
public:
    explicit StreamController(std::atomic<bool> &run) : m_run(run) {
        // Open the video capture
        // Open the default camera (TODO: Can change the camera source, for example to an RTSP
        // stream)
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

        while (m_run) {
            // Sleep so that we match the camera frame rate
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepDurationMs));

            // Grab a frame from the frame buffer, discard the return value
            // This will discard built up frames in the buffer to ensure we only process the latest
            // frame to remove any delay
            m_mtx.lock();
            m_cap.grab();
            m_mtx.unlock();
        }
    }

    // Returns true if a frame was grabber
    bool grabFrame(cv::Mat &frame) {
        const std::lock_guard<std::mutex> lock(m_mtx);
        return m_cap.retrieve(frame);
    }

private:
    std::unique_ptr<std::thread> m_rtspThread = nullptr;
    std::mutex m_mtx;
    std::atomic<bool> &m_run;
    cv::VideoCapture m_cap;
};

int main() {
    std::atomic<bool> run{true};
    StreamController streamController(run);

    // Start by specifying the configuration options to be used.
    // Can choose to use default configuration options if preferred by calling the default SDK
    // constructor. Learn more about configuration options here:
    // https://reference.trueface.ai/cpp/dev/latest/usage/general.html
    ConfigurationOptions options;
    // The face recognition model to use. Balances accuracy and speed.
    options.frModel = FacialRecognitionModel::TFV5_2;
    // The object detection model to use.
    options.objModel = ObjectDetectionModel::FAST;
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
    // By default, the SDK uses lazy initialization, meaning modules are only initialized when they
    // are first used (on first inference). This is done so that modules which are not used do not
    // load their models into memory, and hence do not utilize memory. The downside to this is that
    // the first inference will be much slower as the model file is being decrypted and loaded into
    // memory. Therefore, if you know you will use a module, choose to pre-initialize the module,
    // which reads the model file into memory in the SDK constructor.
    InitializeModule initializeModule;
    initializeModule.objectDetector = true;
    options.initializeModule = initializeModule;

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
    options.gpuOptions.objectDetectorGPUOptions = moduleOptions;

    SDK tfSdk(options);

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense(TRUEFACE_TOKEN);
    if (!isValid) {
        std::cout << "Error: the provided license is invalid\n";
        return -1;
    }

    while (run) {
        // Grab the latest frame from the video stream
        cv::Mat frame;
        auto shouldProcess = streamController.grabFrame(frame);
        if (!shouldProcess) {
            continue;
        }

        // Set the image using the capture frame buffer
        TFImage img;
        auto errorCode =
            tfSdk.preprocessImage(frame.data, frame.cols, frame.rows, ColorCode::bgr, img);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "There was an error setting the image\n";
            return -1;
        }

        // Detect the objects in the frame
        std::vector<BoundingBox> bboxVec;
        tfSdk.detectObjects(img, bboxVec);

        // Annotate the frame
        errorCode = tfSdk.drawObjectLabels(img, bboxVec);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << errorCode << std::endl;
            return -1;
        }

        // Now create a CV image from the frame
        cv::Mat annotated(img->getHeight(), img->getWidth(), CV_8UC3, img->getData());
        cv::cvtColor(annotated, annotated, cv::COLOR_RGB2BGR);

        cv::imshow("frame", annotated);

        if (cv::waitKey(1) == 27) {
            run = false;
            break; // stop capturing by pressing ESC
        }
    }

    return 0;
}