#include "benchmark.h"
#include "stopwatch.h"

#include "tf_data_types.h"
#include "tf_sdk.h"

#include <fstream>
#include <iostream>

using namespace Trueface;

const std::string benchmarkName{"Preprocess image"};

void benchmarkPreprocessImage(const SDKFactory& sdkFactory, BenchmarkParams params, ObservationList& observations) {
    // Initialize the SDK
    auto options = sdkFactory.createBasicConfiguration();
    auto tfSdk = sdkFactory.createSDK(options);
    const std::string imgPath = "../images/headshot.jpg";

    // First run the benchmark for an image on disk
    // Run once to ensure everything works
    TFImage img;
    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            auto errorcode = tfSdk.preprocessImage(imgPath, img);
            if (errorcode != ErrorCode::NO_ERROR) {
                std::cout << "Unable to preprocess the image" << std::endl;
                return;
            }
        }
    }

    // Time the preprocessImage function
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < params.numIterations; ++i) {
        TFImage newImg;
        tfSdk.preprocessImage(imgPath, img);
    }
    auto totalTime = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    auto avgTime = totalTime / params.numIterations;

    std::cout << "Average time preprocessImage JPG image from disk (" << img->getWidth() << "x" << img->getHeight() << "): "
              << avgTime << " ms | " << params.numIterations << " iterations" << std::endl;

    observations.emplace_back(tfSdk.getVersion(), sdkFactory.isGpuEnabled(), benchmarkName, "JPG from disk", "Average Time", params, avgTime);

    // Now repeat with encoded image in memory
    std::ifstream file(imgPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cout << "Unable to load the image" << std::endl;
        return;
    }

    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            auto errorcode = tfSdk.preprocessImage(buffer, img);
            if (errorcode != ErrorCode::NO_ERROR) {
                std::cout << "Unable to preprocess the image" << std::endl;
                return;
            }
        }
    }

    // Time the preprocessImage function
    preciseStopwatch stopwatch1;
    for (size_t i = 0; i < params.numIterations; ++i) {
        TFImage newImg;
        tfSdk.preprocessImage(buffer, newImg);
    }
    totalTime = stopwatch1.elapsedTime<float, std::chrono::milliseconds>();
    avgTime = totalTime / params.numIterations;

    std::cout << "Average time preprocessImage encoded JPG image in memory (" << img->getWidth() << "x" << img->getHeight() << "): "
              << avgTime << " ms | " << params.numIterations << " iterations" << std::endl;

    observations.emplace_back(tfSdk.getVersion(), sdkFactory.isGpuEnabled(), benchmarkName, "encoded JPG in memory", "Average Time", params, avgTime);

    // Now repeat with already decoded imgages (ex. you grab an image from your video stream).
    TFImage newImg;
    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            auto errorCode = tfSdk.preprocessImage(img->getData(), img->getWidth(), img->getHeight(), ColorCode::rgb, newImg);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to preprocess image" << std::endl;
                return;
            }
        }
    }

    preciseStopwatch stopwatch2;
    for (size_t i = 0; i < params.numIterations; ++i) {
        tfSdk.preprocessImage(img->getData(), img->getWidth(), img->getHeight(), ColorCode::rgb, newImg);
    }
    totalTime = stopwatch2.elapsedTime<float, std::chrono::milliseconds>();
    avgTime = totalTime / params.numIterations;

    std::cout << "Average time preprocessImage RGB pixel array in memory (" << img->getWidth() << "x" << img->getHeight() << "): "
              << avgTime << " ms | " << params.numIterations << " iterations" << std::endl;

    observations.emplace_back(tfSdk.getVersion(), sdkFactory.isGpuEnabled(), benchmarkName, "RGB pixels array in memory", "Average Time", params, avgTime);
}