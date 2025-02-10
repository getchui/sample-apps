//
//  SDKManager.swift
//  isamples
//
//  Created by adel boussaken on 11/4/2023.
//

import Foundation

// SDKManager is a singleton class for managing the SDK.
class SDKManager {
    // Shared instance of SDKManager for global access.
    static let shared = SDKManager()

    static let collectionName: String = "DEMO"

    var sdk: TFSDK!
    var options: TFConfigurationOptions!
    private var modelFilesInfo: [(name: String, version: String)] = [] // Store model info

    private init() {
        initSDK()
    }

    // Method to get the writable database path.
    private func getWritableDatabasePath() -> String {
        // Get the documents directory.
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        let documentsDirectory = paths[0]
        // Create a path for the database file in the documents directory.
        let dbPath = documentsDirectory.appendingPathComponent("demo.db").path
        return dbPath
    }

    // MARK: - SDK Initialization
    private func initSDK() {
        // Configure the SDK options.
        configureSDKOptions()
        // Instantiate the SDK with the configured options.
        sdk = TFSDK(configurationOptions: options)
        // Set the SDK license.
        setSDKLicense()
        // Create a database connection and load the collection.
        sdk.createDatabaseConnection(getWritableDatabasePath())
        sdk.createLoadCollection(SDKManager.collectionName)

        print("📱 SDK Version: \(sdk.getVersion() ?? "Unknown")") // Log SDK Version

        if let resourcePath = Bundle.main.resourcePath {
            let fileManager = FileManager.default
            do {
                let contents = try fileManager.contentsOfDirectory(atPath: resourcePath)
                let modelFiles = contents.filter { $0.hasSuffix(".enc") }

                if !modelFiles.isEmpty {
                    print("📂 Model files found in bundle:")
                    for file in modelFiles {
                        let version = extractVersionFromName(fileName: file)
                        print("  ➡️ \(file) | \(version)")
                        modelFilesInfo.append((name: file, version: version)) // Store model info
                    }
                } else {
                    print("⚠️ No model files (.enc) found in bundle.")
                }

            } catch {
                print("❌ Error reading contents of resource path: \(error)")
            }
        } else {
            print("❌ Error: Bundle resource path not found.")
        }
    }

    private func extractVersionFromName(fileName: String) -> String {
        let nameWithoutExtension = fileName.replacingOccurrences(of: ".trueface.v2.enc", with: "")
        let components = nameWithoutExtension.components(separatedBy: "_")
        var modelNameComponents: [String] = []
        var version = "N/A"

        let wordsToRemove = ["detector", "recognition", "detection"] // Words to shorten model names

        for component in components.reversed() {
            if component.hasPrefix("v") {
                let versionNumber = component.dropFirst().trimmingCharacters(in: .letters.inverted)
                if !versionNumber.isEmpty {
                    version = versionNumber
                    break
                }
            }
            modelNameComponents.insert(component, at: 0)
        }

        // Shorten model name by removing keywords
        var shortModelNameComponents: [String] = []
        for component in modelNameComponents {
            let lowercasedComponent = component.lowercased()
            if !wordsToRemove.contains(lowercasedComponent) {
                shortModelNameComponents.append(component)
            }
        }


        let formattedModelName = shortModelNameComponents.map { $0.capitalized }.joined(separator: " ")
        if version != "N/A" {
            return formattedModelName + " V" + version
        } else {
            return formattedModelName
        }
    }


    // Method to configure the SDK options.
    private func configureSDKOptions() {
        options = TFConfigurationOptions()
        options.useCoreML = true
        // options.smallestFaceHeight = 120
        // options.initializeModule.faceRecognizer = true
        options.frModel = LITE_V2
        options.modelsPath = Bundle.main.resourcePath
        options.dbms = SQLITE
    }

    // Method to configure the SDK options.
    private func setSDKLicense() {
        sdk.setLicense("...")
        print("🔑 SDK License valid: \(sdk.isLicensed())")
    }

    // --- Helper functions for AboutViewController ---
    func getSDKVersion() -> String {
        return sdk.getVersion() ?? "Unknown"
    }

    func isSDKLicensed() -> Bool {
        return sdk.isLicensed()
    }

    func getModelListText() -> String {
        var modelListText = ""
        for modelInfo in modelFilesInfo {
            modelListText += "  ➡️ \(modelInfo.version)\n"
        }
        if modelListText.isEmpty {
            return "  ⚠️ No model files found."
        }
        return modelListText
    }
    // --- End Helper functions for AboutViewController ---
}
