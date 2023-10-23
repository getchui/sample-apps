#pragma once

#include "sdkfactory.h"

#include <vector>

// fwd declarations
namespace Trueface {
    enum class FacialRecognitionModel;
    enum class ObjectDetectionModel;
} // namespace Trueface

struct BenchmarkParams {
    bool doWarmup;
    int numWarmup;
    unsigned int batchSize;
    unsigned int numIterations;
};

struct Observation {
    Observation(std::string v, bool gpuEnabled, std::string b, std::string bt, std::string m, BenchmarkParams p, float t)
        : version{v}, isGpuEnabled{gpuEnabled}, benchmark{b}, benchmarkSubType{bt}, measurement{m}, params{p}, timeInMs{t} {}

    std::string version;
    bool isGpuEnabled;
    std::string benchmark;
    std::string benchmarkSubType;
    std::string measurement;
    BenchmarkParams params;
    float timeInMs;
};

using ObservationList = std::vector<Observation>;

void benchmarkFaceRecognition(const SDKFactory&, Trueface::FacialRecognitionModel, BenchmarkParams, ObservationList&);
void benchmarkObjectDetection(const SDKFactory&, Trueface::ObjectDetectionModel objModel, BenchmarkParams, ObservationList&);
void benchmarkFaceLandmarkDetection(const SDKFactory&, BenchmarkParams, ObservationList&);
void benchmarkDetailedLandmarkDetection(const SDKFactory&, BenchmarkParams, ObservationList&);
void benchmarkPreprocessImage(const SDKFactory&, BenchmarkParams, ObservationList&);
void benchmarkMaskDetection(const SDKFactory&, BenchmarkParams, ObservationList&);
void benchmarkBlinkDetection(const SDKFactory&, BenchmarkParams, ObservationList&);
void benchmarkSpoofDetection(const SDKFactory&, BenchmarkParams, ObservationList&);
void benchmarkHeadOrientation(const SDKFactory&, BenchmarkParams, ObservationList&);
void benchmarkFaceImageBlurDetection(const SDKFactory&, BenchmarkParams, ObservationList&);
void benchmarkFaceImageOrientationDetection(const SDKFactory&, BenchmarkParams, ObservationList&);
void benchmarkGlassesDetection(const SDKFactory&, BenchmarkParams, ObservationList&);
