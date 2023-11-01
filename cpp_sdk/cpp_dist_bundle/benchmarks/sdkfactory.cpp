#include "sdkfactory.h"

#include "tf_data_types.h"
#include "tf_sdk.h"

#include <iostream>

using namespace Trueface;

namespace Trueface {
namespace Benchmarks {

SDKFactory::SDKFactory(const GPUOptions &gpuOptions)
    : m_gpuOptions{gpuOptions}, m_modelsPath{"./"}, m_license{TRUEFACE_TOKEN} {
    auto modelsPath = std::getenv("MODELS_PATH");
    if (modelsPath) {
        m_modelsPath = modelsPath;
    }
}

SDK SDKFactory::createSDK(const ConfigurationOptions &options) const {
    SDK tfSdk(options);
    bool valid = tfSdk.setLicense(m_license);
    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        exit(EXIT_FAILURE);
    }

    return tfSdk;
}

ConfigurationOptions SDKFactory::createBasicConfiguration() const {
    ConfigurationOptions options;
    options.modelsPath = m_modelsPath;
    options.gpuOptions = m_gpuOptions;
    return options;
}

GPUOptions SDKFactory::createGPUOptions(bool enableGPU, unsigned int deviceIndex,
                                        int32_t maxBatchSize, int32_t optBatchSize) {
    GPUOptions gpuOptions;
    gpuOptions.enableGPU = enableGPU;
    gpuOptions.deviceIndex = deviceIndex;

    GPUModuleOptions gpuModuleOptions;
    gpuModuleOptions.precision = Precision::FP16;

    gpuModuleOptions.maxBatchSize = maxBatchSize;
    gpuModuleOptions.optBatchSize = optBatchSize;

    gpuOptions.faceDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceRecognizerGPUOptions = gpuModuleOptions;
    gpuOptions.maskDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.objectDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceLandmarkDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceOrientationDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.faceBlurDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.spoofDetectorGPUOptions = gpuModuleOptions;
    gpuOptions.blinkDetectorGPUOptions = gpuModuleOptions;

    return gpuOptions;
}

bool SDKFactory::isGpuEnabled() const { return m_gpuOptions.enableGPU; }

} // namespace Benchmarks
} // namespace Trueface