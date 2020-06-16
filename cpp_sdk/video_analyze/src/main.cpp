#include <chrono>

#include <opencv2/opencv.hpp>

#include "tf_sdk.h"
#include "tf_data_types.h"

void clearConsole()
{
    std::cout << "\x1B[2J\x1B[H";
}

const char* getCheckmark(bool checkmark) {
	return (const char*) checkmark ? "✓" : "✗";
}

int main(int argc, char *argv[])
{

    Trueface::SDK tfSdk;

    // TODO: replace <LICENSE_CODE> with your license code.
    const auto isValid = tfSdk.setLicense("<LICENSE_CODE>");
    if (!isValid)
    {
        std::cout << "Error: the provided license is invalid\n";
        return -1;
    }

    cv::VideoCapture capture;
    if (argc == 2)
    {
        printf("using media\n");
        if (!capture.open(argv[1]))
        {
            printf("failed to open media %s\n", argv[1]);
            return 0;
        }
    }
    else
    {
        printf("using webcam\n");
        if (!capture.open(0))
        {
            printf("failed to open webcam\n");
            return 0;
        }
    }

    if (!capture.isOpened())
	{
		throw "Error when reading steam_avi";
	}

    std::chrono::milliseconds ms(20000); // 20 seconds
    std::chrono::time_point<std::chrono::system_clock> end;
    end = std::chrono::system_clock::now() + ms;

    while (true)
    {
        clearConsole();

        // Read the frame from the VideoCapture source
        cv::Mat frame;
        capture >> frame;
        if (frame.empty())
        {
            break; // End of video stream
        }

        // Set the image using the capture frame buffer
        auto errorCode = tfSdk.setImage(frame.data, frame.cols, frame.rows, Trueface::ColorCode::bgr);
        if (errorCode != Trueface::ErrorCode::NO_ERROR)
        {
            std::cout << "There was an error setting the image\n";
            return -1;
        }

        Trueface::FrameAnalyzeResult result;
        tfSdk.analyzeVideoFrame(result);

        printf("result->isAllPointsChecked %s\n", getCheckmark(result.isAllPointsChecked));
		printf("\n\n");

		printf("result->brightnessScore %f\n", result.brightnessScore);
		printf("result->isBright %s\n", getCheckmark(result.isBright));
		printf("\n\n");

		printf("result->qualityScore %f\n", result.qualityScore);
		printf("result->isQuality %s\n", getCheckmark(result.isQuality));
		printf("\n\n");

		printf("result->eyeDistance %f\n", result.eyeDistance);
		printf("result->isEyeDistanceGood %s\n", getCheckmark(result.isEyeDistanceGood));
		printf("\n\n");

		printf("result->rightEyeWinkScore %f\n", result.rightEyeWinkScore);
		printf("result->leftEyeWinkScore %f\n", result.leftEyeWinkScore);
		printf("result->isBlinking %s\n", getCheckmark(result.isBlinking));
		printf("result->blinksCount %d\n", result.blinksCount);
		printf("\n\n");

		printf("result->yaw: %f\n", result.yaw);
		printf("result->pitch: %f\n", result.pitch);
		printf("result->roll: %f\n", result.roll);
		printf("face direction: 0 no data, 1 stright, 2 left, 3 right\n");
		printf("face direction: %s\n", getCheckmark(result.faceDirection == 1));
		printf("\n\n");

		printf("result->spoofScore %f\n", result.spoofScore);
		printf("result->isSpoofed %s\n", getCheckmark(!result.isSpoofed));
		printf("\n\n");

        if (result.isAllPointsChecked) {
            // success flow
        }

        if (std::chrono::system_clock::now() < end) {
            // failed flow
        }

        cv::imshow("frame", frame);

        if (cv::waitKey(10) == 27)
        {
            break; // stop capturing by pressing ESC
        }
    }

    return 0;
}
