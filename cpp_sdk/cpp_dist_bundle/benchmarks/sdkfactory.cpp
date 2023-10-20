#include "sdkfactory.h"

#include "tf_data_types.h"
#include "tf_sdk.h"

#include <iostream>

using namespace Trueface;

SDKFactory::SDKFactory(const GPUOptions& gpuOptions)
    : gpuOptions_{gpuOptions}, modelsPath_{"./"}, license_{TRUEFACE_TOKEN} {
    auto modelsPath = std::getenv("MODELS_PATH");
    if (modelsPath) {
        modelsPath_ = modelsPath;
    }
}

SDK SDKFactory::createSDK(ConfigurationOptions& options) const {
    SDK tfSdk(options);
    bool valid = tfSdk.setLicense(license_);
    if (!valid) {
        std::cout << "Error: the provided license is invalid." << std::endl;
        exit (EXIT_FAILURE);
    }

    return tfSdk;
}

ConfigurationOptions SDKFactory::createBasicConfiguration() const {
    ConfigurationOptions options;
    options.modelsPath = modelsPath_;
    options.gpuOptions = gpuOptions_;
    return options;
}