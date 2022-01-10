// The following code runs speed benchmarks for the 1:N identification module

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "tf_sdk.h"

typedef std::chrono::high_resolution_clock Clock;


int main() {
    // TODO: Replace with your license
    const std::string license = "${LICENSE_CODE}";

    Trueface::ConfigurationOptions options;
    options.frModel = Trueface::FacialRecognitionModel::TFV5;
    options.dbms = Trueface::DatabaseManagementSystem::NONE;

    // TODO modify the following option to test with / without vector compression
    options.frVectorCompression = true;

    if (options.frVectorCompression) {
        std::cout << "~~~~~~~~~~~~~~~ Vector compression enabled ~~~~~~~~~~~~~~~~"  << std::endl;
    }


    Trueface::SDK sdk(options);
    bool valid = sdk.setLicense(license);
    if (!valid) {
        std::cout << "License not valid" << std::endl;
        return -1;
    }

    // Populate our collection with the following distractor templates
    std::vector<std::pair<std::string, std::string>> imagePairList = {
            {"../../images/tom_cruise_1.jpg", "tom cruise"},
            {"../../images/tom_cruise_2.jpg", "tom cruise"},
            {"../../images/tom_cruise_3.jpg", "tom cruise"},
            {"../images/headshot.jpg", "anon"}
    };

    std::vector<std::pair<std::string, Trueface::Faceprint>> dataVec;

    // Generate a template for each of the images
    std::cout << "\nGenerating templates" << std::endl;
    for (const auto& imagePair: imagePairList) {
        auto ret = sdk.setImage(imagePair.first);
        if (ret != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Unable to set image: " << imagePair.first << std::endl;
            return -1;
        }

        Trueface::Faceprint faceprint;
        bool foundFace;
        ret = sdk.getLargestFaceFeatureVector(faceprint, foundFace);

        if (ret != Trueface::ErrorCode::NO_ERROR || !foundFace) {
            std::cout << "Unable to detect face and extract features" << std::endl;
            return -1;
        }

        dataVec.emplace_back(imagePair.second, std::move(faceprint));
    }

    // Prepare the probe vector and the match vector
    auto ret = sdk.setImage("../../images/brad_pitt_1.jpg");
    if (ret != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to set image" << std::endl;
        return -1;
    }

    bool foundFace;
    Trueface::Faceprint probe;
    ret = sdk.getLargestFaceFeatureVector(probe, foundFace);
    if (ret != Trueface::ErrorCode::NO_ERROR || !foundFace) {
        std::cout << "Unable to detect face and extract features" << std::endl;
        return -1;
    }

    ret = sdk.setImage("../../images/brad_pitt_2.jpg");
    if (ret != Trueface::ErrorCode::NO_ERROR) {
        std::cout << "Unable to set image" << std::endl;
        return -1;
    }

    Trueface::Faceprint matchTemplate;
    ret = sdk.getLargestFaceFeatureVector(matchTemplate, foundFace);
    if (ret != Trueface::ErrorCode::NO_ERROR || !foundFace) {
        std::cout << "Unable to detect face and extract features" << std::endl;
        return -1;
    }

    // Collection sizes to test
    std::vector<size_t> collectionSizes {
        1000,
        10000,
        100000,
        1000000
    };

    // Populate the collections
    for (const auto& collectionSize: collectionSizes) {
        std::cout << "Populating collection with " << collectionSize << " templates" << std::endl;

        // We are using the Trueface::DatabaseManagementSystem::NONE so the collection will not persist
        ret = sdk.createLoadCollection("temp_collection");
        if (ret != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Error creating collection" << std::endl;
            return -1;
        }

        // Enroll the templates
        // Note: we are enrolling collectionSize - 1 because we will add the matchTemplate at the end.
        for (size_t i = 0; i < collectionSize; ++i) {

            const auto &data = dataVec[i % dataVec.size()];
            std::string UUID;

            // Enroll the template and the identity
            ret = sdk.enrollFaceprint(data.second, data.first, UUID);
            if (ret != Trueface::ErrorCode::NO_ERROR) {
                std::cout << "Unable to enroll template" << std::endl;
                return -1;
            }
        }

        // Finally enroll the match template
        std::string UUID;
        ret = sdk.enrollFaceprint(matchTemplate, "Brad Pitt", UUID);
        if (ret != Trueface::ErrorCode::NO_ERROR) {
            std::cout << "Unable to enroll template" << std::endl;
            return -1;
        }

        // Run the timing tests
        size_t numIterations = 1000;
        if (collectionSize >= 100000) {
            numIterations = 100;
        }

        Trueface::Candidate candidate;
        bool found;

        auto t1 = Clock::now();
        for (size_t i = 0; i < numIterations; ++i) {
            sdk.identifyTopCandidate(probe, candidate, found);
        }
        auto t2 = Clock::now();

        if (found) {
            std::cout << "Found match: " << candidate.identity << std::endl;
        }

        double avgTime = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / numIterations;
        std::cout << "[Identify] Collection size: " << collectionSize << " templates, Avg time: " << avgTime << " milliseconds ("
                  << std::to_string(numIterations) << " iterations)" << std::endl;

        // Now run batch identification
        std::vector<Trueface::Faceprint> probeFaceprints;
        for (size_t i = 0; i < 100; ++i) {
            probeFaceprints.push_back(probe);
        }

        std::vector<Trueface::Candidate> candidates;
        std::vector<bool> foundCandidates;

        numIterations /= 10;

        t1 = Clock::now();
        for (size_t i = 0; i < numIterations; ++i) {
            sdk.batchIdentifyTopCandidate(probeFaceprints, candidates, foundCandidates);
        }
        t2 = Clock::now();

        avgTime = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / numIterations / probeFaceprints.size();
        std::cout << "[Batch Identify] Collection size: " << collectionSize << " templates, Avg time: " << avgTime << " milliseconds ("
                  << std::to_string(numIterations) << " iterations, batch size = " << probeFaceprints.size() << ")\n" << std::endl;



    }

    return 0;
}
















