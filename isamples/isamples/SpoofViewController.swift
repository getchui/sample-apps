//
//  SecondViewController.swift
//  isamples
//
//  Created by adel boussaken on 11/4/2023.
//

import UIKit

// This class inherits from FaceDetectionViewController and provides custom behavior for the "SpoofViewController".
class SpoofViewController: FaceDetectionViewController {

    // --- State tracking for logging ---
    private var wasFaceDetectedInPreviousFrame: Bool = false
    // --- End state tracking for logging ---

    override func viewDidLoad() {
        super.viewDidLoad()
        print("🎬 SpoofViewController viewDidLoad - Starting Spoof Detection View")
        // Add custom code for your view controller here

        // Create a label to display face detection results
        let label = UILabel(frame: CGRect(x: 0, y: 0, width: view.frame.width, height: 100))
        label.textAlignment = .center
        label.font = UIFont.boldSystemFont(ofSize: 20.0)
        label.numberOfLines = 0
        label.center = view.center
        view.addSubview(label)
        print("🏷️ Spoof result label created and added to view")

        // Create a guide rectangle to help users position their face
        let guideRect = UIView(frame: CGRect(x: 0, y: 0, width: 739, height: 552))
        guideRect.layer.borderWidth = 2.0
        guideRect.layer.borderColor = UIColor.cyan.cgColor
        guideRect.center = view.center
        view.addSubview(guideRect)
        print(" guide rectangle created and added to view")

        // A list of error labels to display in case of any errors
        let errorLabels = ["NO ERROR", "INVALID LICENSE", "FILE READ FAIL", "UNSUPPORTED IMAGE FORMAT", "UNSUPPORTED MODEL", "NO FACE IN FRAME", "FAILED", "COLLECTION_CREATION_ERROR", "DATABASE_CONNECTION_ERROR", "ENROLLMENT_ERROR", "MAX_COLLECTION_SIZE_EXCEEDED","NO RECORD FOUND", "NO COLLECTION FOUND", "COLLECTION DELETION ERROR", "EXTREME FACE ANGLE", "FACE TOO CLOSE", "FACE TOO FAR", "FACE TOO SMALL", "FACE NOT CENTERED", "EYES CLOSED", "MASK DETECTED", "TOO DARK", "TOO BRIGHT"]

        // Handle the event when a face is detected
        self.onFaceDetected = { [weak self] (sdk, tfImage, face) in
            guard let self = self else { return } // Prevent retain cycle

            let faceDetectedThisFrame = true // Current frame face detection status

            if !self.wasFaceDetectedInPreviousFrame {
                print("😊 Face detected - Starting Spoof Detection (State Change)") // State change log for face detection
            }
            // Get the spoof prediction for the detected face
            let spoof: TFSpoofResult = self.sdk.detectSpoof(in: tfImage, with: face, threshold: 0.5)
            print("🔮 Spoof prediction result obtained")

            // If there's an error, display the error message
            if spoof.errorCode != NO_ERROR {
                DispatchQueue.main.async {
                    let index = Int(spoof.errorCode.rawValue)
                    let error = errorLabels[index]
                    label.text = "Error: \(error)"
                    label.textColor = UIColor.white
                    print("❌ Spoof Error: \(error)")
                }
                return
            }

            // If the face is real, display "Real Face" and the score
            if spoof.label == REAL {
                DispatchQueue.main.async {
                    label.text = "Real Face\nScore: \(spoof.score)"
                    label.textColor = UIColor.green
                    print("✅ Real Face detected - Score: \(spoof.score)")
                }
            }

            // If the face is fake, display "Fake Face" and the score
            if spoof.label == FAKE {
                DispatchQueue.main.async {
                    label.text = "Fake Face\nScore: \(spoof.score)"
                    label.textColor = UIColor.red
                    print("🔴 Fake Face detected - Score: \(spoof.score)")
                }
            }
            self.wasFaceDetectedInPreviousFrame = faceDetectedThisFrame // Update face detection state
        }

        // Handle the event when no face is detected
        self.onFaceNotDetected = { [weak self] sdk, TFImage in
            guard let self = self else { return } // Prevent retain cycle

            let faceDetectedThisFrame = false // Current frame face detection status

            if self.wasFaceDetectedInPreviousFrame { // State change check - Face just disappeared
                DispatchQueue.main.async {
                    label.text = ""
                    print("😔 No face detected - Clearing spoof result label (State Change)") // State change log - reduced spam
                }
            } else {
                // Optionally log for continuous no-face (less important)
                // print("😔 No face detected - Spoof label remains clear")
            }
            self.wasFaceDetectedInPreviousFrame = faceDetectedThisFrame // Update face detection state
        }
        print("🎭 Face detection and spoof handlers set")
    }

    // Resume the camera when the view appears
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        print("WillAppear: Resuming camera for Spoof Detection")
        resumeCamera()
        print("✅ Camera resumed for Spoof Detection")
        wasFaceDetectedInPreviousFrame = false // Reset state when view appears
    }

    // Pause the camera when the view disappears
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        print("WillDisappear: Pausing camera and uninitializing Spoof modules")
        pauseCamera()
        print("⏸️ Camera paused for Spoof Detection")
        print("♻️ Uninitializing Spoof, Mask, Blink, and Landmark modules")
        let uninitializeModule = TFUninitializeModule()
        uninitializeModule.activeSpoof = true
        uninitializeModule.passiveSpoof = true
        uninitializeModule.maskDetector = true
        uninitializeModule.blinkDetector = true
        uninitializeModule.landmarkDetector = true
        self.sdk.uninitializeModule(uninitializeModule)
        print("🎬 SpoofViewController viewWillDisappear completed")
        wasFaceDetectedInPreviousFrame = false // Reset state when view disappears
    }

}
