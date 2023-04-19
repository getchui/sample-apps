//
//  ViewController.swift
//  isamples
//
//  Created by adel boussaken on 10/4/2023.
//

import UIKit
import AVFoundation

// This class inherits from FaceDetectionViewController and provides custom behavior.
class ViewController: FaceDetectionViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Add custom code for your view controller here
        
        // Handle the event when a face is detected.
        self.onFaceDetected = { (sdk, tfImage, face) in
            // Add custom code for handling detected face (e.g., print coordinates)
            // print("face found \(face.topLeft.x) / \(face.topLeft.y)")
        }
        
        // Handle the event when no face is detected.
        self.onFaceNotDetected = { (sdk, tfImage) in
            // Add custom code for handling the absence of a face in the current frame
            // For example, hide or remove UI elements related to face detection
        }
        
        let buttonBackground = UIView()
        buttonBackground.backgroundColor = UIColor.black.withAlphaComponent(0.4)
        buttonBackground.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(buttonBackground)
        
        let stackView = UIStackView()
        stackView.axis = .horizontal
        stackView.distribution = .fillEqually
        stackView.spacing = 10
        stackView.translatesAutoresizingMaskIntoConstraints = false
        
        let backgroundView = UIView()
        backgroundView.backgroundColor = UIColor.black.withAlphaComponent(0.7) // Darker background
        backgroundView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(stackView)
        
        let buttonTitles = ["Spoof", "Face Recognize", "Enroll", "Manage People"]
        
        for title in buttonTitles {
            let button = UIButton(type: .system)
            button.setTitle(title, for: .normal)
            button.tintColor = .white
            button.addTarget(self, action: #selector(buttonTapped), for: .touchUpInside)
            stackView.addArrangedSubview(button)
        }
        
        NSLayoutConstraint.activate([
            stackView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 8),
            stackView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -8),
            stackView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            
            buttonBackground.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            buttonBackground.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            buttonBackground.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
            buttonBackground.topAnchor.constraint(equalTo: stackView.topAnchor, constant: -8)
        ])
    }
    
    @objc func buttonTapped(_ sender: UIButton) {
        if let buttonTitle = sender.title(for: .normal) {
            switch buttonTitle {
            case "Spoof":
                let spoofViewController = SpoofViewController()
                navigationController?.pushViewController(spoofViewController, animated: true)
            case "Face Recognize":
                let frfViewController = FRViewController()
                navigationController?.pushViewController(frfViewController, animated: true)
            case "Enroll":
                let enrollViewController = EnrollViewController()
                navigationController?.pushViewController(enrollViewController, animated: true)
            case "Manage People":
                let managePeopleViewController = ManagePeopleViewController()
                navigationController?.pushViewController(managePeopleViewController, animated: true)
            default:
                break
            }
        }
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        resumeCamera()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        pauseCamera()
    }
    
}
