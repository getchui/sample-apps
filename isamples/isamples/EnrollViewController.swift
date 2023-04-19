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

    override func viewDidLoad() {
        super.viewDidLoad()
        // Add custom code for your view controller here
        
        // Create and configure the camera button
        cameraButton.setImage(UIImage(systemName: "camera"), for: .normal)
        cameraButton.tintColor = .white
        cameraButton.isEnabled = false
        cameraButton.addTarget(self, action: #selector(cameraButtonTapped), for: .touchUpInside)
        cameraButton.translatesAutoresizingMaskIntoConstraints = false
        
        view.addSubview(cameraButton)
        
        NSLayoutConstraint.activate([
            cameraButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            cameraButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20)
        ])
        
        // Enable the camera button and store the TFImage and face when a face is detected
        self.onFaceDetected = { (sdk, tfImage, face) in
            DispatchQueue.main.async {
                self.cameraButton.isEnabled = true
            }
            self.tfImage = tfImage
            self.face = face
        }
        
        // Disable the camera button when no face is detected
        self.onFaceNotDetected = { (sdk, tfImage) in
            DispatchQueue.main.async {
                self.cameraButton.isEnabled = false
            }
        }
    }
    
    // Enroll the face and associated name when the camera button is tapped
    @objc func cameraButtonTapped() {
        let alertController = UIAlertController(title: "Enroll Face", message: "Enter the name of the person", preferredStyle: .alert)
        
        alertController.addTextField { (textField) in
            textField.placeholder = "Name"
        }
        
        let enrollAction = UIAlertAction(title: "Enroll", style: .default) { [weak self] _ in
            guard let self = self,
                  let textField = alertController.textFields?.first,
                  let nameOfPerson = textField.text,
                  !nameOfPerson.isEmpty,
                  let sdk = self.sdk,
                  let tfImage = self.tfImage,
                  let face = self.face else { return }
            
            let faceprint = sdk.getFaceFeatureVector(tfImage, face)
            sdk.enrollFaceprint(faceprint, nameOfPerson)
        }
        
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        
        alertController.addAction(enrollAction)
        alertController.addAction(cancelAction)
        
        present(alertController, animated: true, completion: nil)
    }
    
    // Resume the camera when the view appears
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        resumeCamera()
    }
    
    // Pause the camera when the view disappears
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        pauseCamera()
    }
}
