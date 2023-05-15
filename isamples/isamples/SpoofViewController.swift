//
//  SecondViewController.swift
//  isamples
//
//  Created by adel boussaken on 11/4/2023.
//

import UIKit

// This class inherits from FaceDetectionViewController and provides custom behavior for the "SpoofViewController".
class SpoofViewController: FaceDetectionViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Add custom code for your view controller here
        
        // Create a label to display face detection results
        let label = UILabel(frame: CGRect(x: 0, y: 0, width: view.frame.width, height: 100))
        label.textAlignment = .center
        label.font = UIFont.boldSystemFont(ofSize: 20.0)
        label.numberOfLines = 0
        label.center = view.center
        view.addSubview(label)

        // Create a guide rectangle to help users position their face
        let guideRect = UIView(frame: CGRect(x: 0, y: 0, width: 739, height: 552))
        guideRect.layer.borderWidth = 2.0
        guideRect.layer.borderColor = UIColor.cyan.cgColor
        guideRect.center = view.center
        view.addSubview(guideRect)
        
        // A list of error labels to display in case of any errors
        let errorLabels = ["NO ERROR", "INVALID LICENSE", "FILE READ FAIL", "UNSUPPORTED IMAGE FORMAT", "UNSUPPORTED MODEL", "NO FACE IN FRAME", "FAILED", "COLLECTION_CREATION_ERROR", "DATABASE_CONNECTION_ERROR", "ENROLLMENT_ERROR", "MAX_COLLECTION_SIZE_EXCEEDED","NO RECORD FOUND", "NO COLLECTION FOUND", "COLLECTION DELETION ERROR", "EXTREME FACE ANGLE", "FACE TOO CLOSE", "FACE TOO FAR", "FACE TOO SMALL", "FACE NOT CENTERED", "EYES CLOSED", "MASK DETECTED", "TOO DARK", "TOO BRIGHT"]
        
        // Handle the event when a face is detected
        self.onFaceDetected = { (sdk, tfImage, face) in
            // Get the spoof prediction for the detected face
            let spoof: TFSpoofPrediction = self.sdk.detectSpoof(tfImage, face, 0.5)
            
            // If there's an error, display the error message
            if spoof.errorCode != NO_ERROR {
                DispatchQueue.main.async {
                    let index = Int(spoof.errorCode.rawValue)
                    let error = errorLabels[index]
                    label.text = "Error: \(error)"
                    label.textColor = UIColor.white
                }
                return
            }
            
            // If the face is real, display "Real Face" and the score
            if spoof.label == REAL {
                DispatchQueue.main.async {
                    label.text = "Real Face\nScore: \(spoof.score)"
                    label.textColor = UIColor.green
                }
            }
            
            // If the face is fake, display "Fake Face" and the score
            if spoof.label == FAKE {
                DispatchQueue.main.async {
                    label.text = "Fake Face\nScore: \(spoof.score)"
                    label.textColor = UIColor.red
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
        print("uninitializeModule")
        let uninitializeModule = TFUninitializeModule()
        uninitializeModule.activeSpoof = true
        uninitializeModule.passiveSpoof = true
        uninitializeModule.maskDetector = true
        uninitializeModule.liveness = true
        uninitializeModule.landmarkDetector = true
        self.sdk.uninitializeModule(uninitializeModule)
    }
    
}
