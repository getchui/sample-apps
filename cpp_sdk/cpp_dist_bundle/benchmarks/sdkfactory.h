#pragma once

#include <string>

namespace Trueface {
    struct GPUOptions;
    struct ConfigurationOptions;
    class SDK;
}

namespace Trueface {
namespace Benchmarks {
class SDKFactory
{
public:
    SDKFactory(const Trueface::GPUOptions& gpuOptions);

    Trueface::SDK createSDK(const Trueface::ConfigurationOptions& options) const;
    Trueface::ConfigurationOptions createBasicConfiguration() const;
    static Trueface::GPUOptions createGPUOptions(bool enableGPU, unsigned int deviceIndex,
        int32_t maxBatchSize, int32_t optBatchSize);

    bool isGpuEnabled() const;

private:
    const Trueface::GPUOptions& m_gpuOptions;
    std::string m_modelsPath;
    std::string m_license;
};

} // namespace Benchmarks
} // namespace Trueface