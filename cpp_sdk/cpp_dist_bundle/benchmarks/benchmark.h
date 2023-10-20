#pragma once

#include "sdkfactory.h"
#include "tf_data_types.h"

struct BenchmarkParams {
    bool doWarmup;
    int numWarmup;
    unsigned int batchSize;
    unsigned int numIterations;
};

void benchmarkFaceRecognition(Trueface::FacialRecognitionModel, const SDKFactory&, BenchmarkParams);
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
