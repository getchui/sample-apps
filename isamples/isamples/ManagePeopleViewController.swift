//
//  ManagePeopleViewController.swift
//  isamples
//
//  Created by adel boussaken on 11/4/2023.
//

import UIKit

// This class allows managing the enrolled identities
class ManagePeopleViewController: UIViewController, UITableViewDataSource, UITableViewDelegate {

    var tableView: UITableView!
    var identities: [TFCollectionIdentities] = []
    let sdk: TFSDK = SDKManager.shared.sdk
    let collectionName = "DEMO"

    override func viewDidLoad() {
        super.viewDidLoad()
        print("🎬 ManagePeopleViewController viewDidLoad - Starting Manage People View")

        // Setup the table view
        tableView = UITableView(frame: view.bounds, style: .plain)
        tableView.dataSource = self
        tableView.delegate = self
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "PersonCell")
        view.addSubview(tableView)
        print("🏓 TableView created and configured")

        // Retrieve the identities from the collection
        print("📚 Fetching identities from collection: \(SDKManager.collectionName)")
        if let result = sdk.getCollectionIdentities(SDKManager.collectionName),
          let identityList = result.collectionIdentities as? [TFCollectionIdentities] {
           self.identities = identityList
            print("✅ Retrieved \(identities.count) identities from collection")
        } else {
            print("⚠️ Failed to retrieve identities from collection or collection is empty")
        }

        // Set navigation bar title
        title = "Manage People"
        print("🏷️ Navigation bar title set to: Manage People")
        print("🎬 ManagePeopleViewController viewDidLoad completed")
    }

    // UITableViewDataSource methods
    func numberOfSections(in tableView: UITableView) -> Int {
        if identities.isEmpty {
            // Display a label when there are no people in the collection, but still return 1 section
            let noDataLabel = UILabel(frame: CGRect(x: 0, y: 0, width: tableView.bounds.size.width, height: tableView.bounds.size.height))
            noDataLabel.text = "No people found"
            noDataLabel.textColor = .gray
            noDataLabel.textAlignment = .center
            tableView.backgroundView = noDataLabel
            tableView.separatorStyle = .none
        } else {
            tableView.backgroundView = nil
            tableView.separatorStyle = .singleLine
        }
        print("📊 Number of sections: Always returning 1 section") // Log change
        return 1 // **Corrected: Always return 1 section**
    }

    // UITableViewDataSource methods
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        print("🔢 Number of rows in section: \(identities.count)")
        return identities.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "PersonCell", for: indexPath)
        let identity = identities[indexPath.row]
        let text = "\(identity.identity ?? "") (\(identity.uuid ?? ""))"
        cell.textLabel?.text = text
        print("➡️ Cell created for row: \(indexPath.row), Identity: \(identity.identity ?? "N/A"), UUID: \(identity.uuid ?? "N/A")")
        return cell
    }

    // UITableViewDelegate methods
    // **NEW: Implement swipe-to-delete**
    func tableView(_ tableView: UITableView, commit editingStyle: UITableViewCell.EditingStyle, forRowAt indexPath: IndexPath) {
        if editingStyle == .delete {
            let identityToRemove = identities[indexPath.row]
            print("🔥 Swipe-to-delete detected for identity: \(identityToRemove.identity ?? "N/A")")

            print("🔥 Removing identity: \(identityToRemove.identity ?? "N/A") from collection: \(SDKManager.collectionName)")
            let _ = self.sdk.remove(byIdentity: identityToRemove.identity, collectionName: SDKManager.collectionName)

            self.identities.remove(at: indexPath.row) // Remove from data source
            tableView.deleteRows(at: [indexPath], with: .fade) // Animate row deletion
            print("✅ Identity removed via swipe-to-delete and TableView updated")
        }
    }

    // **UPDATED: tableView(_:didSelectRowAt:) - Now does nothing, tap selection removed**
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)
        print("👆 Row selected at index: \(indexPath.row) - Tap selection now does nothing")
        // No action on tap anymore, deletion is handled by swipe-to-delete
    }

    // **Optional: Customize delete button text (for localization)**
    func tableView(_ tableView: UITableView, titleForDeleteConfirmationButtonForRowAt indexPath: IndexPath) -> String? {
        return "Remove" // You can localize this string
    }
}
