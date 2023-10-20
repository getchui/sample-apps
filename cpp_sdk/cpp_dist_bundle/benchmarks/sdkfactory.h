#pragma once

#include "tf_data_types.h"
#include "tf_sdk.h"

#include <iostream>
#include <string>

class SDKFactory
{
public:
    SDKFactory(const Trueface::GPUOptions& gpuOptions) : gpuOptions_{gpuOptions}, modelsPath_{"./"}, license_{TRUEFACE_TOKEN} {
        auto modelsPath = std::getenv("MODELS_PATH");
        if (modelsPath) {
            modelsPath_ = modelsPath;
        }
    }

    Trueface::SDK createSDK(Trueface::ConfigurationOptions& options) const {
        Trueface::SDK tfSdk(options);
        bool valid = tfSdk.setLicense(license_);
        if (!valid) {
            std::cout << "Error: the provided license is invalid." << std::endl;
            exit (EXIT_FAILURE);
        }

        return tfSdk;
    }

    Trueface::ConfigurationOptions createBasicConfiguration() const {
        Trueface::ConfigurationOptions options;
        options.modelsPath = modelsPath_;
        options.gpuOptions = gpuOptions_;
        return options;
    }

private:

    const Trueface::GPUOptions& gpuOptions_;
    std::string modelsPath_;
    std::string license_;
};