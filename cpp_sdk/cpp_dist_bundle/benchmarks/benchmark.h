#pragma once

#include "sdkfactory.h"
#include "observation.h"

// fwd declarations
namespace Trueface {
    enum class FacialRecognitionModel;
    enum class ObjectDetectionModel;
} // namespace Trueface

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