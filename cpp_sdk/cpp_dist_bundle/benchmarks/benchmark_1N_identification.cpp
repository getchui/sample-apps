// The following code runs speed benchmarks for the 1:N identification module
#include "observation.h"
#include "sdkfactory.h"
#include "stopwatch.h"

#include "tf_sdk.h"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

using namespace Trueface;

int main() {
    auto gpuOptions = GPUOptions(false);
    auto sdkFactory = Benchmarks::SDKFactory(gpuOptions);

    auto options = sdkFactory.createBasicConfiguration();
    options.frModel = FacialRecognitionModel::TFV7;
    options.dbms = DatabaseManagementSystem::NONE;

    // TODO modify the following option to test with / without vector compression
    options.frVectorCompression = true;

    if (options.frVectorCompression) {
        std::cout << "~~~~~~~~~~~~~~~ Vector compression enabled ~~~~~~~~~~~~~~~~" << std::endl;
    }

    auto sdk = sdkFactory.createSDK(options);

    // Populate our collection with the following distractor templates
    std::vector<std::pair<std::string, std::string>> imagePairList = {
        {"../../images/tom_cruise_1.jpg", "tom cruise"},
        {"../../images/tom_cruise_2.jpg", "tom cruise"},
        {"../../images/tom_cruise_3.jpg", "tom cruise"},
        {"../images/headshot.jpg", "anon"}};

    std::vector<std::pair<std::string, Faceprint>> dataVec;

    // Generate a template for each of the images
    std::cout << "\nGenerating templates" << std::endl;
    for (const auto &imagePair : imagePairList) {
        TFImage img;
        auto ret = sdk.preprocessImage(imagePair.first, img);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Unable to set image: " << imagePair.first << std::endl;
            return -1;
        }

        Faceprint faceprint;
        bool foundFace;
        ret = sdk.getLargestFaceFeatureVector(img, faceprint, foundFace);

        if (ret != ErrorCode::NO_ERROR || !foundFace) {
            std::cout << "Unable to detect face and extract features" << std::endl;
            return -1;
        }

        dataVec.emplace_back(imagePair.second, std::move(faceprint));
    }

    // Prepare the probe vector and the match vector
    TFImage img;
    auto ret = sdk.preprocessImage("../../images/brad_pitt_1.jpg", img);
    if (ret != ErrorCode::NO_ERROR) {
        std::cout << "Unable to set image" << std::endl;
        return -1;
    }

    bool foundFace;
    Faceprint probe;
    ret = sdk.getLargestFaceFeatureVector(img, probe, foundFace);
    if (ret != ErrorCode::NO_ERROR || !foundFace) {
        std::cout << "Unable to detect face and extract features" << std::endl;
        return -1;
    }

    ret = sdk.preprocessImage("../../images/brad_pitt_2.jpg", img);
    if (ret != ErrorCode::NO_ERROR) {
        std::cout << "Unable to set image" << std::endl;
        return -1;
    }

    Faceprint matchTemplate;
    ret = sdk.getLargestFaceFeatureVector(img, matchTemplate, foundFace);
    if (ret != ErrorCode::NO_ERROR || !foundFace) {
        std::cout << "Unable to detect face and extract features" << std::endl;
        return -1;
    }

    // Collection sizes to test
    std::vector<size_t> collectionSizes{1000, 10000, 100000, 1000000};

    auto observations = Benchmarks::ObservationList();
    // Populate the collections
    for (const auto &collectionSize : collectionSizes) {
        std::cout << "Populating collection with " << collectionSize << " templates" << std::endl;

        // We are using the DatabaseManagementSystem::NONE so the collection will not persist
        ret = sdk.createLoadCollection("temp_collection");
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Error creating collection" << std::endl;
            return -1;
        }

        // Enroll the templates
        // Note: we are enrolling collectionSize - 1 because we will add the matchTemplate at the
        // end.
        for (size_t i = 0; i < collectionSize; ++i) {

            const auto &data = dataVec[i % dataVec.size()];
            std::string UUID;

            // Enroll the template and the identity
            ret = sdk.enrollFaceprint(data.second, data.first, UUID);
            if (ret != ErrorCode::NO_ERROR) {
                std::cout << "Unable to enroll template" << std::endl;
                return -1;
            }
        }

        // Finally enroll the match template
        std::string UUID;
        ret = sdk.enrollFaceprint(matchTemplate, "Brad Pitt", UUID);
        if (ret != ErrorCode::NO_ERROR) {
            std::cout << "Unable to enroll template" << std::endl;
            return -1;
        }

        // Run the timing tests
        auto parameters = Benchmarks::Parameters{false, 0, 1, 1000};
        if (collectionSize >= 100000) {
            parameters.numIterations = 100;
        }

        Candidate candidate;
        bool found;

        auto times = std::vector<float>();
        times.reserve(parameters.numIterations);
        for (size_t i = 0; i < parameters.numIterations; ++i) {
            auto stopwatch = preciseStopwatch();
            sdk.identifyTopCandidate(probe, candidate, found);
            times.emplace_back(stopwatch.elapsedTime<float, std::chrono::nanoseconds>());
        }

        if (found) {
            std::cout << "Found match: " << candidate.identity << std::endl;
        }

        std::string benchmarkName = "1 to N identification search";
        std::string benchmarkSubType = "(" + std::to_string(collectionSize) + ") TFV7";
        observations.emplace_back(sdk.getVersion(), sdkFactory.isGpuEnabled(), benchmarkName,
                                  benchmarkSubType, parameters, times, 0.0f);

        // Now run batch identification
        std::vector<Faceprint> probeFaceprints;
        for (size_t i = 0; i < 100; ++i) {
            probeFaceprints.push_back(probe);
        }

        std::vector<Candidate> candidates;
        std::vector<bool> foundCandidates;

        parameters.numWarmup /= 10;
        parameters.batchSize = probeFaceprints.size();

        times.clear();
        for (size_t i = 0; i < parameters.numIterations; ++i) {
            auto stopwatch = preciseStopwatch();
            sdk.batchIdentifyTopCandidate(probeFaceprints, candidates, foundCandidates);
            times.emplace_back(stopwatch.elapsedTime<float, std::chrono::nanoseconds>());
        }

        benchmarkName = "1 to N batch identification search";
        observations.emplace_back(sdk.getVersion(), sdkFactory.isGpuEnabled(), benchmarkName,
                                  benchmarkSubType, parameters, times, 0.0f);
    }

    auto csvWriter = Benchmarks::ObservationCSVWriter("benchmarks.csv");
    csvWriter.write(observations);

    return 0;
}
