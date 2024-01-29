#include "memory_high_water_mark.h"
#include "observation.h"
#include "sdkfactory.h"
#include "stopwatch.h"

#include "tf_data_types.h"
#include "tf_sdk.h"

#include <fstream>
#include <iostream>

using namespace Trueface;
using namespace Trueface::Benchmarks;

const std::string benchmarkName{"Preprocess image"};

void benchmarkPreprocessImage(const SDKFactory &sdkFactory, Parameters params,
                              ObservationList &observations) {
    // Baseline memory reading
    auto memoryTracker = MemoryHighWaterMarkTracker();

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
    std::vector<float> times;
    times.reserve(params.numIterations);
    for (size_t i = 0; i < params.numIterations; ++i) {
        preciseStopwatch stopwatch;
        tfSdk.preprocessImage(imgPath, img);
        times.emplace_back(stopwatch.elapsedTime<float, std::chrono::nanoseconds>());
    }

    observations.emplace_back(tfSdk.getVersion(), sdkFactory.isGpuEnabled(), benchmarkName,
                              "JPG from disk", params, times,
                              memoryTracker.getDifferenceFromBaseline());

    // baseline memory reading
    memoryTracker.resetVmHighWaterMark();

    // Now repeat with encoded image in memory
    std::ifstream file(imgPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
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
    times.clear();
    for (size_t i = 0; i < params.numIterations; ++i) {
        preciseStopwatch stopwatch;
        tfSdk.preprocessImage(buffer, img);
        times.emplace_back(stopwatch.elapsedTime<float, std::chrono::nanoseconds>());
    }

    observations.emplace_back(tfSdk.getVersion(), sdkFactory.isGpuEnabled(), benchmarkName,
                              "encoded JPG in memory", params, times,
                              memoryTracker.getDifferenceFromBaseline());

    // Baseline memory reading
    memoryTracker.resetVmHighWaterMark();

    // Now repeat with already decoded imgages (ex. you grab an image from your video stream).
    TFImage newImg;
    if (params.doWarmup) {
        for (int i = 0; i < params.numWarmup; ++i) {
            auto errorCode = tfSdk.preprocessImage(img->getData(), img->getWidth(),
                                                   img->getHeight(), ColorCode::rgb, newImg);
            if (errorCode != ErrorCode::NO_ERROR) {
                std::cout << "Error: Unable to preprocess image" << std::endl;
                return;
            }
        }
    }

    times.clear();
    for (size_t i = 0; i < params.numIterations; ++i) {
        preciseStopwatch stopwatch;
        tfSdk.preprocessImage(img->getData(), img->getWidth(), img->getHeight(), ColorCode::rgb,
                              newImg);
        times.emplace_back(stopwatch.elapsedTime<float, std::chrono::nanoseconds>());
    }

    observations.emplace_back(tfSdk.getVersion(), sdkFactory.isGpuEnabled(), benchmarkName,
                              "RGB pixels array in memory", params, times,
                              memoryTracker.getDifferenceFromBaseline());
}