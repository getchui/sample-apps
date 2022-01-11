#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <thread>
#include <mutex>
#include <memory>
#include <atomic>
#include <chrono>
#include <cmath>

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

// Utility function for drawing the label on our image
void setLabel(cv::Mat& im, const std::string label, const cv::Point & oldOrigin, const cv::Scalar& color) {
    cv::Point origin(oldOrigin.x - 2, oldOrigin.y - 10);
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    // Can change scale and thickness to change label size
    const double scale = 0.8;
    const int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, font, scale, thickness, &baseline);
    cv::rectangle(im, origin + cv::Point(0, baseline), origin + cv::Point(text.width, -text.height), color, cv::FILLED);
    cv::putText(im, label, origin, font, scale, CV_RGB(0,0,0), thickness, cv::LINE_AA);
}


int main() {
    std::atomic<bool> run {true};
    StreamController streamController(run);

    // TODO: Select a threshold for your application using the ROC curves
    // https://docs.trueface.ai/ROC-Curves-d47d2730cf0a44afacb39aae0ed1b45a
    const float threshold = 0.3;

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
    options.smallestFaceHeight = 40;
    // The path specifying the directory where the model files have been downloaded
    options.modelsPath = "./";
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
    initializeModule.faceRecognizer = true;
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
    const auto isValid = tfSdk.setLicense("<LICENSE_CODE>");
    if (!isValid) {
        std::cout << "Error: the provided license is invalid\n";
        return -1;
    }

    // Create a new database
    // This step is not required if using DatabaseManagementSystem::NONE
    const std::string databaseName = "myDatabase.db";
    auto errorCode = tfSdk.createDatabaseConnection(databaseName);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: unable to create database\n";
        return -1;
    }

    // Create a collection
    const std::string collectioName = "myCollection";
    errorCode = tfSdk.createLoadCollection(collectioName);

    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: unable to create collection\n";
        return -1;
    }

    // Load the image / images we want to enroll
    errorCode = tfSdk.setImage("../../../../images/armstrong/armstrong1.jpg");
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: unable to read image\n";
        return -1;
    }

    // Detect the face in the image
    FaceBoxAndLandmarks faceBoxAndLandmarks;
    bool faceDetected;
    errorCode = tfSdk.detectLargestFace(faceBoxAndLandmarks, faceDetected);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "There was an error with the call to detectLargestFace\n";
        return -1;
    }

    if (!faceDetected) {
        std::cout << "Unable to detect face\n";
        return -1;
    }

    // We want to only enroll high quality images into the database / collection
    // Therefore, ensure that the face height is at least 100px
    auto faceHeight = faceBoxAndLandmarks.bottomRight.y - faceBoxAndLandmarks.topLeft.y;
    std::cout << "Face height: " << faceHeight << std::endl;
    if (faceHeight < 100) {
        std::cout << "The face is too small in the image for a high quality enrollment." << std::endl;
        return -1;
    }

    // Get the aligned face chip so that we can compute the image quality
    uint8_t alignedImage[37632];
    errorCode = tfSdk.extractAlignedFace(faceBoxAndLandmarks, alignedImage);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "There was an error extracting the aligned face\n";
        return -1;
    }

    float quality;
    errorCode = tfSdk.estimateFaceImageQuality(alignedImage, quality);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute image quality\n";
        return -1;
    }

    // Ensure the image quality is above a threshold (TODO: adjust this threshold based on your use case).
    // Once again, we only want to enroll high quality images into our collection
    std::cout << "Face quality: " << quality << std::endl;
    if (quality < 0.8) {
        std::cout << "Please choose a higher quality enrollment image\n";
        return -1;
    }

    // As a final check, we can check the orientation of the head and ensure that it is facing forward
    // To see the effect of yaw and pitch on the match score, refer to: https://reference.trueface.ai/cpp/dev/latest/usage/face.html#_CPPv4N8Trueface3SDK23estimateHeadOrientationERK19FaceBoxAndLandmarksRfRfRf
    float yaw, pitch, roll;
    errorCode = tfSdk.estimateHeadOrientation(faceBoxAndLandmarks, yaw, pitch, roll);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Unable to compute head orientation\n";
        return -1;
    }

    std::cout << "Yaw: " << yaw * 180 / M_PI  << ", Pitch: " << pitch * 180 / M_PI << ", Roll: " << roll * 180 / M_PI << " degrees" << std::endl;

    // TODO: Can filter out images with extreme yaw and pitch here

    // Generate the enrollment template
    Faceprint enrollmentFaceprint;
    errorCode = tfSdk.getFaceFeatureVector(alignedImage, enrollmentFaceprint);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to generate template\n";
        return -1;
    }

    // Add the enrollment template to our collection
    // Any data that is added to the collection will persist after the application is terminated
    std::string UUID;
    errorCode = tfSdk.enrollFaceprint(enrollmentFaceprint, "Armstrong", UUID);
    if (errorCode != ErrorCode::NO_ERROR) {
        std::cout << "Error: Unable to enroll template\n";
        return -1;
    }

    // Can add other template pairs to the collection here...

    while(run) {
        // Grab the latest frame from the video stream
        cv::Mat frame;
        auto shouldProcess = streamController.grabFrame(frame);
        if (!shouldProcess) {
            continue;
        }

        // Set the image using the capture frame buffer
        errorCode = tfSdk.setImage(frame.data, frame.cols, frame.rows, ColorCode::bgr);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "There was an error setting the image\n";
            return -1;
        }

        // Get all the bounding boxes
        std::vector<FaceBoxAndLandmarks> bboxVec;
        tfSdk.detectFaces(bboxVec);

        // For each bounding box, get the face feature vector
        std::vector<Faceprint> faceprints;
        faceprints.reserve(bboxVec.size());

        for (const auto &bbox: bboxVec) {
            // Get the face feature vector
            Faceprint tmpFaceprint;
            tfSdk.getFaceFeatureVector(bbox, tmpFaceprint);
            faceprints.emplace_back(std::move(tmpFaceprint));
        }

        // Run batch identification on the faceprints
        std::vector<bool> found;
        std::vector<Candidate> candidates;
        tfSdk.batchIdentifyTopCandidate(faceprints, candidates, found, threshold);

        // If the identity was found, draw the identity label
        // If the identity was not found, blur the face

        for (size_t i = 0; i < found.size(); ++i) {
            const auto& bbox = bboxVec[i];
            const auto& candidate = candidates[i];

            if (found[i]) {
                // The identity was found, draw the identity label
                cv::Point topLeft(bbox.topLeft.x, bbox.topLeft.y);
                cv::Point bottomRight(bbox.bottomRight.x, bbox.bottomRight.y);
                cv::Scalar color(0, 255, 0);
                cv::rectangle(frame, topLeft, bottomRight, color, 2);
                setLabel(frame, candidate.identity, topLeft, color);
            } else {
                // The identity was not found, blur the face
                const auto faceWidth = bbox.bottomRight.x - bbox.topLeft.x;
                const auto blurSize = static_cast<int>(faceWidth / 8);

                // The identity was not found, blur the face
                cv::Rect blurRect(bbox.topLeft.x, bbox.topLeft.y, bbox.bottomRight.x - bbox.topLeft.x, bbox.bottomRight.y - bbox.topLeft.y);
                cv::blur(frame(blurRect), frame(blurRect), cv::Size(blurSize, blurSize));

                // Draw white rectangle around face
                cv::Point topLeft(bbox.topLeft.x, bbox.topLeft.y);
                cv::Point bottomRight(bbox.bottomRight.x, bbox.bottomRight.y);
                cv::Scalar color(255, 255, 255);
                cv::rectangle(frame, topLeft, bottomRight, color, 2);
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