#pragma once

#include <string>

namespace Trueface {
    struct GPUOptions;
    struct ConfigurationOptions;
    class SDK;
}

class SDKFactory
{
public:
    SDKFactory(const Trueface::GPUOptions& gpuOptions);

    Trueface::SDK createSDK(Trueface::ConfigurationOptions& options) const;
    Trueface::ConfigurationOptions createBasicConfiguration() const;
    static Trueface::GPUOptions createGPUOptions(bool enableGPU, unsigned int deviceIndex,
        int32_t maxBatchSize, int32_t optBatchSize);

    bool isGpuEnabled() const;

private:
    const Trueface::GPUOptions& gpuOptions_;
    std::string modelsPath_;
    std::string license_;
};