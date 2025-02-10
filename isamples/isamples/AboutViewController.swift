//
//  AboutViewController.swift
//  isamples
//
//  Created by adel boussaken on 10/2/2025.
//

import UIKit

// Create a simple AboutViewController
class AboutViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .white // Or any background color you prefer
        title = "About App"

        let textView = UITextView()
        textView.translatesAutoresizingMaskIntoConstraints = false
        textView.isEditable = false
        textView.font = UIFont.systemFont(ofSize: 14) // Slightly smaller font for more info
        view.addSubview(textView)

        NSLayoutConstraint.activate([
            textView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            textView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            textView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            textView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20)
        ])

        // Get SDK information from SDKManager
        let sdkVersion = SDKManager.shared.getSDKVersion()
        let isLicensed = SDKManager.shared.isSDKLicensed()
        let modelList = SDKManager.shared.getModelListText()

        // Format the about text
        var aboutText = "This is a sample app demonstrating face recognition using Trueface SDK.\n\n"
        aboutText += "SDK Version: \(sdkVersion)\n"
        aboutText += "License Status: \(isLicensed ? "Licensed" : "Not Licensed")\n\n"
        aboutText += "Available Models:\n\(modelList)"

        textView.text = aboutText
    }
}
