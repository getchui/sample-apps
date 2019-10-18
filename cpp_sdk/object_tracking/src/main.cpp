#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

#include "tf_sdk.h"
#include "tf_data_types.h"

// class used for object tracking
class Tracker {
public:
    // Perform the tracking
    void update(const std::vector<Trueface::BoundingBox>& bboxes);
private:
    // Register a new object
    void registerObj(const cv::Point& centroid);
    // Deregister an object
    void deregisterObj(const int objId);

    // Max allowable number of frames that an object can disappear
    const int MAX_FRAMES_DISAPPEARED = 50;

    int m_objId = 0;
    // Keep track of the object IDs and their centroids
    std::unordered_map<int, cv::Point> m_objsMap;
    // Keep track of the number of consecutive frames that the object has disappeared
    std::unordered_map<int, int> m_disappearedObjsMap;
};


// Utility function for drawing the label on our image
    void setLabel(cv::Mat& im, const std::string label, const cv::Point & origin) {
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.4;
    const int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, font, scale, thickness, &baseline);
    cv::rectangle(im, origin + cv::Point(0, baseline), origin + cv::Point(text.width, -text.height), CV_RGB(0,0,0), cv::FILLED);
    cv::putText(im, label, origin, font, scale, CV_RGB(255,255,255), thickness, 8);
}


int main() {
    Trueface::SDK tfSdk;

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense("<LICENSE_CODE>");
    if (!isValid) {
        std::cout << "Error: the provided license is invalid\n";
        return -1;
    }

    cv::VideoCapture cap;
    // Open the default camera, use something different from 0 otherwise
    if (!cap.open(0)) {
        std::cout << "Unable to open video capture\n";
        return -1;
    }

    while(true) {
        // Read the frame from the VideoCapture source
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            break; // End of video stream
        }

        // Set the image using the capture frame buffer
        auto errorCode = tfSdk.setImage(frame.data, frame.cols, frame.rows, Trueface::ColorCode::bgr);
        if (errorCode != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "There was an error setting the image\n";
            return -1;
        }

        // Detect the objects in the frame
        std::vector<Trueface::BoundingBox> bboxVec;
        tfSdk.detectObjects(bboxVec);

        // Display the bounding boxes and labels for the detected objects
        for (const auto& bbox: bboxVec) {
            // TODO: Can use the bbox.probability to filter results if desired

            // Draw a rectangle using the top left and bottom right coordinates of the bounding box
            cv::Point topLeft(bbox.topLeft.x, bbox.topLeft.y);
            cv::Point bottomRight(bbox.topLeft.x + bbox.width, bbox.topLeft.y + bbox.height);
            cv::rectangle(frame, topLeft, bottomRight, cv::Scalar(255, 0, 0));

            // Convert the object label to a string
            // Draw the string on the frame
            const auto label = tfSdk.getObjectLabelString(bbox.label);
            setLabel(frame, label, topLeft);
        }

        cv::imshow("frame", frame);

        if (cv::waitKey(10) == 27) {
            break; // stop capturing by pressing ESC
        }
    }

    return 0;
}

void Tracker::registerObj(const cv::Point& centroid) {
        // Register and object using the next available object ID to store the centroid
        m_objsMap[m_objId] = centroid;
        m_disappearedObjsMap[m_objId] = 0;
        ++m_objId;
}

void Tracker::deregisterObj(const int objId) {
        // Deregister an object ID by deleting the object ID from both our maps
        m_objsMap.erase(objId);
        m_disappearedObjsMap.erase(objId);
}

void Tracker::update(const std::vector<Trueface::BoundingBox>& bboxes) {
        if (bboxes.empty()) {
            // Loop over the existing tracked objects and mark them as disappeared
            for (auto& obj : m_disappearedObjsMap) {
                ++obj.second;

                // If the object has disappeared for more than the maximum allowable number
                // of frames, deregister the object
                if (obj.second > MAX_FRAMES_DISAPPEARED) {
                    deregisterObj(obj.first);
                }
            }
        } else {
            // Loop over the bounding boxes and compute the centroid
            std::vector<cv::Point> centroidVec;
            for (const auto& bbox: bboxes) {
                const auto cx = (bbox.topLeft.x + bbox.width) / 2.0;
                const auto cy = (bbox.topLeft.y + bbox.height) / 2.0;
                centroidVec.emplace_back(cv::Point(cx, cy));
            }

            if (m_objsMap.empty()) {
                // If we are currently not tracking any objects, take the input centroids
                // and register each of them
                for (const auto& centroid: centroidVec) {
                    registerObj(centroid);
                }
            } else {
                // Otherwise we are currently tracking objects so we need to try to match
                // the input centroids to existing object centroids

                // Compute the distance between all the registered centroids and input centroids
            }
        }
    }
