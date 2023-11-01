#pragma once

#include "observation.h"
#include "sdkfactory.h"

// fwd declarations
namespace Trueface {
enum class FacialRecognitionModel;
enum class ObjectDetectionModel;
} // namespace Trueface

void benchmarkFaceRecognition(const Trueface::Benchmarks::SDKFactory &,
                              Trueface::FacialRecognitionModel, Trueface::Benchmarks::Parameters,
                              Trueface::Benchmarks::ObservationList &);
void benchmarkObjectDetection(const Trueface::Benchmarks::SDKFactory &,
                              Trueface::ObjectDetectionModel objModel,
                              Trueface::Benchmarks::Parameters,
                              Trueface::Benchmarks::ObservationList &);
void benchmarkFaceLandmarkDetection(const Trueface::Benchmarks::SDKFactory &,
                                    Trueface::Benchmarks::Parameters,
                                    Trueface::Benchmarks::ObservationList &);
void benchmarkDetailedLandmarkDetection(const Trueface::Benchmarks::SDKFactory &,
                                        Trueface::Benchmarks::Parameters,
                                        Trueface::Benchmarks::ObservationList &);
void benchmarkPreprocessImage(const Trueface::Benchmarks::SDKFactory &,
                              Trueface::Benchmarks::Parameters,
                              Trueface::Benchmarks::ObservationList &);
void benchmarkMaskDetection(const Trueface::Benchmarks::SDKFactory &,
                            Trueface::Benchmarks::Parameters,
                            Trueface::Benchmarks::ObservationList &);
void benchmarkBlinkDetection(const Trueface::Benchmarks::SDKFactory &,
                             Trueface::Benchmarks::Parameters,
                             Trueface::Benchmarks::ObservationList &);
void benchmarkSpoofDetection(const Trueface::Benchmarks::SDKFactory &,
                             Trueface::Benchmarks::Parameters,
                             Trueface::Benchmarks::ObservationList &);
void benchmarkHeadOrientation(const Trueface::Benchmarks::SDKFactory &,
                              Trueface::Benchmarks::Parameters,
                              Trueface::Benchmarks::ObservationList &);
void benchmarkFaceImageBlurDetection(const Trueface::Benchmarks::SDKFactory &,
                                     Trueface::Benchmarks::Parameters,
                                     Trueface::Benchmarks::ObservationList &);
void benchmarkFaceImageOrientationDetection(const Trueface::Benchmarks::SDKFactory &,
                                            Trueface::Benchmarks::Parameters,
                                            Trueface::Benchmarks::ObservationList &);
void benchmarkGlassesDetection(const Trueface::Benchmarks::SDKFactory &,
                               Trueface::Benchmarks::Parameters,
                               Trueface::Benchmarks::ObservationList &);