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
    
    var sdk: TFSDK!
    var options: TFConfigurationOptions!
    
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
        sdk.createLoadCollection("DEMO")
    }
    
    // Method to configure the SDK options.
    private func configureSDKOptions() {
        options = TFConfigurationOptions()
        options.useCoreML = true
        options.initializeModule.faceRecognizer = true
        options.initializeModule.faceDetector = true
        options.initializeModule.objectDetector = true
        options.objModel = OBJECT_FAST
        options.fdModel = FACE_FAST
        options.frModel = TFV5_2
        options.modelsPath = Bundle.main.resourcePath
        options.dbms = SQLITE
    }
    
    // Method to configure the SDK options.
    private func setSDKLicense() {
        sdk.setLicense("...")
    }
}
