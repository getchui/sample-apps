//
//  FRViewController.swift
//  isamples
//
//  Created by adel boussaken on 13/4/2023.
//

import Foundation
import UIKit

// This class inherits from FaceDetectionViewController and provides custom behavior for the "FRViewController".
class FRViewController: FaceDetectionViewController {

    // --- State tracking for logging ---
    private var wasFaceDetectedInPreviousFrame: Bool = false
    private var lastRecognizedIdentity: String? // Track last recognized identity
    // --- End state tracking for logging ---

    override func viewDidLoad() {
        super.viewDidLoad()
        print("🎬 FRViewController viewDidLoad - Starting Face Recognition View")

        // Add custom code for your view controller here

        // Create a label to display face recognition results
        let label = UILabel(frame: CGRect(x: 0, y: 0, width: view.frame.width, height: 100))
        label.textAlignment = .center
        label.font = UIFont.boldSystemFont(ofSize: 20.0)
        label.numberOfLines = 0
        label.center = view.center
        view.addSubview(label)
        print("🏷️ Recognition result label created and added to view")

        // Handle the event when a face is detected
        self.onFaceDetected = { [weak self] (sdk, tfImage, face) in
            guard let self = self else { return } // Prevent retain cycle and handle self being nil

            let faceDetectedThisFrame = true // Current frame face detection status
            var currentRecognizedIdentity: String? = nil // To track identity in this frame

            if !self.wasFaceDetectedInPreviousFrame { // Check for state change - Face just appeared
                print("😊 Face detected - Starting Face Recognition process (State Change)") // State change log
            } else {
                // Optionally log continuous face presence, less important for logs
                // print("😊 Face detected - Continuing Face Recognition process")
            }

            // Get the face feature vector for the detected face
            if let faceprint = self.sdk.getFaceFeatureVector(for: tfImage, faceBoxAndLandmarks: face) {
                if !self.wasFaceDetectedInPreviousFrame { // State change log condition
                    print("👤 Faceprint extracted successfully") // State change log
                }
                // Identify the top candidate for the face feature vector
                if let result = self.sdk.identifyTopCandidate(with: faceprint, collectionName: SDKManager.collectionName) {
                    if !self.wasFaceDetectedInPreviousFrame { // State change log condition
                        print("🏆 Top candidate identified (or no candidate found)") // State change log
                    }
                    // Get the confidence score for the identified candidate
                    if (result.candidate == nil)  {
                        print("👤 No candidate found in collection")
                        DispatchQueue.main.async {
                            label.text = "Unknown person"
                            label.textColor = .label // Reset to default color when unknown
                        }
                        currentRecognizedIdentity = "Unknown person" // Track for comparison
                        return
                    }
                    let confidence = String(format: "%.2f", 100 * result.candidate.similarityMeasure)
                    let name = result.candidate.identity ?? "Unknown person" // Default to "Unknown person" if no name

                    // **Update label text and color on every frame**
                    DispatchQueue.main.async {
                        label.text = "\(name) \n Confidence: \(confidence)%"
                        label.textColor = .green // Set text color to green
                    }
                    currentRecognizedIdentity = name // Track for comparison


                } else {
                    // If no candidate is found, display "Unknown person"
                    print("👤 No candidate identified in collection")
                    DispatchQueue.main.async {
                        label.text = "Unknown person"
                        label.textColor = .label // Reset to default color when unknown
                    }
                    currentRecognizedIdentity = "Unknown person" // Track for comparison
                }
            } else {
                print("⚠️ Error: Failed to extract faceprint")
                DispatchQueue.main.async {
                    label.text = "Recognition Error"
                    label.textColor = .red // Indicate error with red color
                }
                currentRecognizedIdentity = "Recognition Error" // Track for comparison (error case)
            }
            self.wasFaceDetectedInPreviousFrame = faceDetectedThisFrame // Update face detection state
            self.lastRecognizedIdentity = currentRecognizedIdentity // Update last recognized identity
        }

        // Handle the event when no face is detected
        self.onFaceNotDetected = { [weak self] sdk, TFImage in
            guard let self = self else { return } // Prevent retain cycle and handle self being nil

            let faceDetectedThisFrame = false // Current frame face detection status

            if self.wasFaceDetectedInPreviousFrame { // Check for state change - Face just disappeared
                print("😔 No face detected - Clearing recognition label (State Change)") // State change log
                DispatchQueue.main.async {
                    label.text = ""
                    label.textColor = .label // Reset to default color when no face
                }
                self.lastRecognizedIdentity = nil // Clear last recognized identity as no face
            } else {
                // Optionally log continuous no face, less important for logs
                // print("😔 No face detected - Recognition label remains clear")
            }
            self.wasFaceDetectedInPreviousFrame = faceDetectedThisFrame // Update previous state
        }
        print("🎭 Face detection and recognition handlers set")
    }

    // Resume the camera when the view appears
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        print("WillAppear: Resuming camera for Face Recognition")
        resumeCamera()
        print("✅ Camera resumed for Face Recognition")
        wasFaceDetectedInPreviousFrame = false // Reset face detection state when view appears
        lastRecognizedIdentity = nil // Reset recognized identity when view appears
    }

    // Pause the camera when the view disappears
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        print("WillDisappear: Pausing camera and uninitializing Face Recognizer module")
        pauseCamera()
        print("⏸️ Camera paused for Face Recognition")
        let uninitializeModule = TFUninitializeModule()
        uninitializeModule.faceRecognizer = true
        self.sdk.uninitializeModule(uninitializeModule)
        print("♻️ Face Recognizer module uninitialized")
        print("🎬 FRViewController viewWillDisappear completed")
        wasFaceDetectedInPreviousFrame = false // Reset face detection state when view disappears
        lastRecognizedIdentity = nil // Reset recognized identity when view disappears
    }

}
