#pragma once

#include "sdkfactory.h"
#include "observation.h"

// fwd declarations
namespace Trueface {
    enum class FacialRecognitionModel;
    enum class ObjectDetectionModel;
} // namespace Trueface

void benchmarkFaceRecognition(const SDKFactory&, Trueface::FacialRecognitionModel, Trueface::Benchmarks::Parameters, Trueface::Benchmarks::ObservationList&);
void benchmarkObjectDetection(const SDKFactory&, Trueface::ObjectDetectionModel objModel, Trueface::Benchmarks::Parameters, Trueface::Benchmarks::ObservationList&);
void benchmarkFaceLandmarkDetection(const SDKFactory&, Trueface::Benchmarks::Parameters, Trueface::Benchmarks::ObservationList&);
void benchmarkDetailedLandmarkDetection(const SDKFactory&, Trueface::Benchmarks::Parameters, Trueface::Benchmarks::ObservationList&);
void benchmarkPreprocessImage(const SDKFactory&, Trueface::Benchmarks::Parameters, Trueface::Benchmarks::ObservationList&);
void benchmarkMaskDetection(const SDKFactory&, Trueface::Benchmarks::Parameters, Trueface::Benchmarks::ObservationList&);
void benchmarkBlinkDetection(const SDKFactory&, Trueface::Benchmarks::Parameters, Trueface::Benchmarks::ObservationList&);
void benchmarkSpoofDetection(const SDKFactory&, Trueface::Benchmarks::Parameters, Trueface::Benchmarks::ObservationList&);
void benchmarkHeadOrientation(const SDKFactory&, Trueface::Benchmarks::Parameters, Trueface::Benchmarks::ObservationList&);
void benchmarkFaceImageBlurDetection(const SDKFactory&, Trueface::Benchmarks::Parameters, Trueface::Benchmarks::ObservationList&);
void benchmarkFaceImageOrientationDetection(const SDKFactory&, Trueface::Benchmarks::Parameters, Trueface::Benchmarks::ObservationList&);
void benchmarkGlassesDetection(const SDKFactory&, Trueface::Benchmarks::Parameters, Trueface::Benchmarks::ObservationList&);