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
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Add custom code for your view controller here
        
        // Create a label to display face recognition results
        let label = UILabel(frame: CGRect(x: 0, y: 0, width: view.frame.width, height: 100))
        label.textAlignment = .center
        label.font = UIFont.boldSystemFont(ofSize: 20.0)
        label.numberOfLines = 0
        label.center = view.center
        view.addSubview(label)
        
        // Handle the event when a face is detected
        self.onFaceDetected = { (sdk, tfImage, face) in
            // Get the face feature vector for the detected face
            if let faceprint = self.sdk.getFaceFeatureVector(tfImage, face) {
                // Identify the top candidate for the face feature vector
                if let candidate = self.sdk.identifyTopCandidate(faceprint) {
                    // Get the confidence score for the identified candidate
                    let confidence = candidate.similarityMeasure

                    // Get the name of the identified candidate
                    if let name = candidate.identity {
                        DispatchQueue.main.async {
                            label.text = "\(name) \n Confidence: \(confidence)"
                        }
                    }
                } else {
                    // If no candidate is found, display "Unknown person"
                    DispatchQueue.main.async {
                        label.text = "Unknown person"
                    }
                }
            }
        }
        
        // Handle the event when no face is detected
        self.onFaceNotDetected = { sdk, TFImage in
            DispatchQueue.main.async {
                label.text = ""
            }
        }
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
        let uninitializeModule = TFUninitializeModule()
        uninitializeModule.faceRecognizer = true
        self.sdk.uninitializeModule(uninitializeModule)
    }

}

