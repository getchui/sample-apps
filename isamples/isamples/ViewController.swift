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
        print("🎬 ViewController viewDidLoad") // Log: View Controller lifecycle start

        // Add custom code for your view controller here

        // Handle the event when a face is detected.
        self.onFaceDetected = { (sdk, tfImage, face) in
            // Add custom code for handling detected face (e.g., print coordinates)
            // print("face found \(face.topLeft.x) / \(face.topLeft.y)")
            // Consider adding a log here if face detection logging is needed, but might be too frequent.
            // print("😊 Face Detected: \(face.topLeft), \(face.bottomRight)")
        }

        // Handle the event when no face is detected.
        self.onFaceNotDetected = { (sdk, tfImage) in
            // Add custom code for handling the absence of a face in the current frame
            // For example, hide or remove UI elements related to face detection
            // Consider adding a log here if no face detection logging is needed, but might be too frequent.
            // print("😔 No Face Detected")
        }
        print("🎭 Face detection handlers set") // Log: Face detection handlers are configured

        let buttonBackground = UIView()
        buttonBackground.backgroundColor = UIColor.black.withAlphaComponent(0.4)
        buttonBackground.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(buttonBackground) // **Important: Add buttonBackground to view**

        let stackView = UIStackView()
        stackView.axis = .horizontal
        stackView.distribution = .fillEqually
        stackView.spacing = 10
        stackView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(stackView) // **Important: Add stackView to view**


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
        print("✅ UI elements configured in viewDidLoad") // Log: UI setup complete

        // Add info button to navigation bar
        let infoButton = UIBarButtonItem(image: UIImage(systemName: "info.circle.fill"), style: .plain, target: self, action: #selector(infoButtonTapped))
        navigationItem.rightBarButtonItem = infoButton
        print("ℹ️ Info button added to navigation bar")
    }

    @objc func buttonTapped(_ sender: UIButton) {
        if let buttonTitle = sender.title(for: .normal) {
            print("Button tapped: \(buttonTitle)") // Log: Button tap event
            switch buttonTitle {
            case "Spoof":
                print("➡️ Navigating to SpoofViewController") // Log: Navigation action
                let spoofViewController = SpoofViewController()
                navigationController?.pushViewController(spoofViewController, animated: false)
            case "Face Recognize":
                print("➡️ Navigating to FRViewController") // Log: Navigation action
                let frfViewController = FRViewController()
                navigationController?.pushViewController(frfViewController, animated: false)
            case "Enroll":
                print("➡️ Navigating to EnrollViewController") // Log: Navigation action
                let enrollViewController = EnrollViewController()
                navigationController?.pushViewController(enrollViewController, animated: false)
            case "Manage People":
                print("➡️ Navigating to ManagePeopleViewController") // Log: Navigation action
                let managePeopleViewController = ManagePeopleViewController()
                navigationController?.pushViewController(managePeopleViewController, animated: false)
            default:
                print("⚠️ Unknown button tapped: \(buttonTitle)") // Log: Unexpected button tap
                break
            }
        }
    }

    @objc func infoButtonTapped() {
        print("ℹ️ Info button tapped") // Log: Info button tap event
        let aboutViewController = AboutViewController()
        print("➡️ Navigating to AboutViewController") // Log: Navigation action
        navigationController?.pushViewController(aboutViewController, animated: true)
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        print("WillAppear: Resuming camera") // Log: View will appear, camera resume intent
        resumeCamera()
        print("✅ Camera resumed") // Log: Camera resume action complete
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        print("WillDisappear: Pausing camera") // Log: View will disappear, camera pause intent
        pauseCamera()
        print("⏸️ Camera paused") // Log: Camera pause action complete
    }

}
