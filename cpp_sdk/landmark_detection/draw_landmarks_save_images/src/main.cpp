#include <opencv2/opencv.hpp>
#include <vector>
#include <experimental/filesystem>

#include "tf_sdk.h"
#include "tf_data_types.h"

namespace fs = std::experimental::filesystem;

// Return a list of all the files in a directory
std::vector<std::string> getFilesInDir(const std::string& path);

int main() {
    Trueface::ConfigurationOptions options;
    options.smallestFaceHeight = 30;
    Trueface::SDK tfSdk(options);

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense("<LICENSE_CODE>");
    if (!isValid) {
        std::cout << "Error: the provided license is invalid\n";
        return -1;
    }

    // Can modify this path to point to another directory with images
    const std::string imageDirPath = "../../../images";
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
        auto errorCode = tfSdk.setImage(image.data, image.cols, image.rows, Trueface::ColorCode::bgr);
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "There was an error setting the image\n";
            return -1;
        }

        // Get the landmark locations
        std::vector<Trueface::FaceBoxAndLandmarks> landmarksVec;
        tfSdk.detectFaces(landmarksVec);

        if (landmarksVec.empty()) {
            std::cout << "Unable to detect face in image: " << imagePath << '\n';
            continue;
        }

        // Display the landmark locations and bounding box on the image
        for (const auto& landmarks: landmarksVec) {
            // Only use the landmarks / bounding box if the score is above 0.90
            if (landmarks.score < 0.90)
                continue;

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