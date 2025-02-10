//
//  EnrollViewController.swift
//  isamples
//
//  Created by adel boussaken on 11/4/2023.
//

import Foundation

import UIKit

// This class inherits from FaceDetectionViewController and provides custom behavior for enrolling faces.
class EnrollViewController: FaceDetectionViewController {
    let cameraButton = UIButton(type: .system)
    var tfImage: TFImage?
    var face: TFFaceBoxAndLandmarks?

    // --- State tracking for logging ---
    private var wasFaceDetectedInPreviousFrame: Bool = false
    // --- End state tracking for logging ---

    override func viewDidLoad() {
        super.viewDidLoad()
        print("🎬 EnrollViewController viewDidLoad - Starting Face Enrollment View")

        // Add custom code for your view controller here

        // Create and configure the camera button
        cameraButton.setImage(UIImage(systemName: "camera"), for: .normal)
        cameraButton.tintColor = .white
        cameraButton.isEnabled = false
        cameraButton.addTarget(self, action: #selector(cameraButtonTapped), for: .touchUpInside)
        cameraButton.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(cameraButton)
        print("📸 Camera button created and added to view")

        NSLayoutConstraint.activate([
            cameraButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            cameraButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20)
        ])
        print("📐 Camera button constraints activated")

        // Enable the camera button and store the TFImage and face when a face is detected
        self.onFaceDetected = { [weak self] (sdk, tfImage, face) in
            guard let self = self else { return } // Prevent retain cycle

            let faceDetectedThisFrame = true // Current frame face detection status

            if !self.wasFaceDetectedInPreviousFrame { // Check for state change - Face just appeared
                DispatchQueue.main.async {
                    self.cameraButton.isEnabled = true
                    print("✅ Face detected - Enabling camera button (State Change)") // State change log
                }
                self.tfImage = tfImage
                self.face = face
                print("😊 Face data (tfImage, face) stored for enrollment (State Change)") // State change log
            } else {
                // Optionally log if you want to track continuous face presence (less important for logs)
                // print("😊 Face data (tfImage, face) updated - Face continuously detected")
            }
            self.wasFaceDetectedInPreviousFrame = faceDetectedThisFrame // Update previous state
        }

        // Disable the camera button when no face is detected
        self.onFaceNotDetected = { [weak self] (sdk, tfImage) in
            guard let self = self else { return } // Prevent retain cycle

            let faceDetectedThisFrame = false // Current frame face detection status

            if self.wasFaceDetectedInPreviousFrame { // Check for state change - Face just disappeared
                DispatchQueue.main.async {
                    self.cameraButton.isEnabled = false
                    print("😔 No face detected - Disabling camera button (State Change)") // State change log
                }
                self.tfImage = nil // Clear potentially old face data
                self.face = nil
                print("😔 Face data cleared - No face detected (State Change)") // State change log
            } else {
                // Optionally log if you want to track continuous no-face presence (less important for logs)
                // print("😔 No face detected - Still no face detected")
            }
            self.wasFaceDetectedInPreviousFrame = faceDetectedThisFrame // Update previous state
        }
        print("🎭 Face detection handlers set for enrollment")
    }

    // Enroll the face and associated name when the camera button is tapped
    @objc func cameraButtonTapped() {
        print("⏺️ Camera button tapped - Initiating face enrollment process")
        let alertController = UIAlertController(title: "Enroll Face", message: "Enter the name of the person", preferredStyle: .alert)

        alertController.addTextField { (textField) in
            textField.placeholder = "Name"
            print("✏️ Text field added to alert for name input")
        }

        let enrollAction = UIAlertAction(title: "Enroll", style: .default) { [weak self] _ in
            print("👤 Enroll action selected in alert")
            guard let self = self,
                  let textField = alertController.textFields?.first,
                  let nameOfPerson = textField.text,
                  !nameOfPerson.isEmpty,
                  let sdk = self.sdk,
                  let tfImage = self.tfImage,
                  let face = self.face else {
                print("⚠️ Enrollment cancelled - Missing required data (self, textField, name, sdk, tfImage, face)")
                return
            }

            let faceprint = sdk.getFaceFeatureVector(for: tfImage, faceBoxAndLandmarks: face)
            if faceprint != nil {
                print("🧬 Faceprint extracted for enrollment")
                sdk.enrollFaceprint(faceprint, withIdentity: nameOfPerson, collectionName: SDKManager.collectionName)
                print("✅ Face enrolled with name: \(nameOfPerson) in collection: \(SDKManager.collectionName)")
            } else {
                print("❌ Error: Faceprint extraction failed, enrollment aborted")
            }
        }

        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel) { _ in
            print("❌ Enrollment cancelled by user")
        }

        alertController.addAction(enrollAction)
        alertController.addAction(cancelAction)
        print("➕ Enroll and Cancel actions added to alert controller")

        present(alertController, animated: true, completion: {
            print("💬 Enrollment alert presented to user")
        })
    }

    // Resume the camera when the view appears
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        print("WillAppear: Resuming camera for Face Enrollment")
        resumeCamera()
        print("✅ Camera resumed for Face Enrollment")
        wasFaceDetectedInPreviousFrame = false // Reset state when view appears
    }

    // Pause the camera when the view disappears
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        print("WillDisappear: Pausing camera for Face Enrollment")
        pauseCamera()
        print("⏸️ Camera paused for Face Enrollment")
        print("🎬 EnrollViewController viewWillDisappear completed")
        wasFaceDetectedInPreviousFrame = false // Reset state when view disappears
    }
}
