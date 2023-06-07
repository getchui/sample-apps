#include <opencv2/opencv.hpp>
#include <vector>
#include <experimental/filesystem>

#include "tf_sdk.h"
#include "tf_data_types.h"

using namespace Trueface;

namespace fs = std::experimental::filesystem;

// Return a list of all the files in a directory
std::vector<std::string> getFilesInDir(const std::string& path);

// Utility function for drawing the label on our image
void setLabel(cv::Mat& im, const std::string label, const cv::Point & origin) {
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.6;
    const int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, font, scale, thickness, &baseline);
    cv::rectangle(im, origin + cv::Point(0, baseline), origin + cv::Point(text.width, -text.height), CV_RGB(0,0,0), cv::FILLED);
    cv::putText(im, label, origin, font, scale, CV_RGB(255,255,255), thickness, cv::LINE_AA);
}

int main() {
    // Start by specifying the configuration options to be used.
    // Can choose to use default configuration options if preferred by calling the default SDK constructor.
    // Learn more about configuration options here: https://reference.trueface.ai/cpp/dev/latest/usage/general.html
    ConfigurationOptions options;
    // The face recognition model to use. Balances accuracy and speed.
    options.frModel = FacialRecognitionModel::TFV5_2;
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

    SDK tfSdk(options);

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense(TRUEFACE_TOKEN);
    if (!isValid) {
        std::cout << "Error: the provided license is invalid\n";
        return -1;
    }

    // Can modify this path to point to another directory with images
    const std::string imageDirPath = "../../../../images";
    const auto imageList = getFilesInDir(imageDirPath);

    int imageNum = 0;

    for(const auto& imagePath: imageList) {
        std::cout << "Loading image: " << ++imageNum << "/" << imageList.size() << '\n';
        auto image = cv::imread(imagePath);

        if (!image.data) {
            // Unable to read image
            continue;
        }

        // OpenCV loads images as BGR
        TFImage img;
        auto errorCode = tfSdk.preprocessImage(image.data, image.cols, image.rows, ColorCode::bgr, img);
        if (errorCode != ErrorCode::NO_ERROR) {
            std::cout << "There was an error setting the image\n";
            return -1;
        }

        // Get the landmark locations
        std::vector<FaceBoxAndLandmarks> landmarksVec;
        tfSdk.detectFaces(img, landmarksVec);

        if (landmarksVec.empty()) {
            std::cout << "Unable to detect face in image: " << imagePath << '\n';
            continue;
        }

        // Display the landmark locations and bounding box on the image
        for (const auto& landmarks: landmarksVec) {

            // Draw a rectangle using the top left and bottom right coordinates of the bounding box
            cv::Point topLeft(landmarks.topLeft.x, landmarks.topLeft.y);
            cv::Point bottomRight(landmarks.bottomRight.x, landmarks.bottomRight.y);
            cv::rectangle(image, topLeft, bottomRight, cv::Scalar(255, 0, 0), 2);

            // Draw the facial landmarks
            // the facial landmark points: left eye, right eye, nose, left mouth corner, right mouth corner
            for (const auto& landmark: landmarks.landmarks) {
                cv::Point p(landmark.x, landmark.y);
                cv::circle(image, p, 1, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
            }

            // Draw the face score onto the face
            setLabel(image, std::to_string(landmarks.score), cv::Point(landmarks.topLeft.x, landmarks.topLeft.y));
        }

        // Write the image to disk
        // my/image/path/armstrong.png
        auto imageName = imagePath.substr(imagePath.find_last_of('/') + 1, imagePath.find_last_of('.') - imagePath.find_last_of('/') - 1);
        auto imageSuffix = imagePath.substr(imagePath.find_last_of('.'));
        auto outputName = imageName + "_landmarks" + imageSuffix;
        cv::imwrite(outputName, image);
    }

    return 0;
}

std::vector<std::string> getFilesInDir(const std::string& path) {
    fs::recursive_directory_iterator iter(path);
    fs::recursive_directory_iterator end;
    std::vector<std::string> listOfFiles;

    while(iter != end) {
        if (!fs::is_directory(iter->path())) {
            listOfFiles.push_back(iter->path().string());
        }
        std::error_code ec;
        iter.increment(ec);
        if (ec) {
            std::string errMsg = "Error While Accessing : " + iter->path().string() + " :: " + ec.message() + '\n';
            throw std::invalid_argument(errMsg);
        }
    }
    return listOfFiles;
}