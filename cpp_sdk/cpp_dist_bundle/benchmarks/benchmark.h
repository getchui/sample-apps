#pragma once

#include "sdkfactory.h"

struct BenchmarkParams {
    bool doWarmup;
    int numWarmup;
    unsigned int batchSize;
    unsigned int numIterations;
};

// fwd declarations
namespace Trueface {
    enum class FacialRecognitionModel;
    enum class ObjectDetectionModel;
} // namespace Trueface

struct Observation {
    std::string benchmark;
    std::string benchmarkSubType;
    BenchmarkParams params;
    float timeInMs;
};

void benchmarkFaceRecognition(const SDKFactory&, Trueface::FacialRecognitionModel, BenchmarkParams);
void benchmarkObjectDetection(const SDKFactory&, Trueface::ObjectDetectionModel objModel, BenchmarkParams);
void benchmarkFaceLandmarkDetection(const SDKFactory&, BenchmarkParams);
void benchmarkDetailedLandmarkDetection(const SDKFactory&, BenchmarkParams);
void benchmarkPreprocessImage(const SDKFactory&, BenchmarkParams);
void benchmarkMaskDetection(const SDKFactory&, BenchmarkParams);
void benchmarkBlinkDetection(const SDKFactory&, BenchmarkParams);
void benchmarkSpoofDetection(const SDKFactory&, BenchmarkParams);
void benchmarkHeadOrientation(const SDKFactory&, BenchmarkParams);
void benchmarkFaceImageBlurDetection(const SDKFactory&, BenchmarkParams);
void benchmarkFaceImageOrientationDetection(const SDKFactory&, BenchmarkParams);
void benchmarkGlassesDetection(const SDKFactory&, BenchmarkParams);
