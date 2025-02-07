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

        // Setup the table view
        tableView = UITableView(frame: view.bounds, style: .plain)
        tableView.dataSource = self
        tableView.delegate = self
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "PersonCell")
        view.addSubview(tableView)
        
        // Retrieve the identities from the collection
        if let result = sdk.getCollectionIdentities(SDKManager.collectionName),
          let identityList = result.collectionIdentities as? [TFCollectionIdentities] {
           self.identities = identityList
        }

        // Set navigation bar title
        title = "Manage People"
    }
    
    // UITableViewDataSource methods
    func numberOfSections(in tableView: UITableView) -> Int {
        if identities.isEmpty {
            // Display a label when there are no people in the collection
            let noDataLabel = UILabel(frame: CGRect(x: 0, y: 0, width: tableView.bounds.size.width, height: tableView.bounds.size.height))
            noDataLabel.text = "No people found"
            noDataLabel.textColor = .gray
            noDataLabel.textAlignment = .center
            tableView.backgroundView = noDataLabel
            tableView.separatorStyle = .none
            return 0
        } else {
            tableView.backgroundView = nil
            tableView.separatorStyle = .singleLine
            return 1
        }
    }

    // UITableViewDataSource methods
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return identities.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "PersonCell", for: indexPath)
        let identity = identities[indexPath.row]
        let text = "\(identity.identity ?? "") (\(identity.uuid ?? ""))"
        cell.textLabel?.text = text
        return cell
    }

    // UITableViewDelegate methods
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)

        let alertController = UIAlertController(title: "Remove Person", message: "Do you want to remove this person?", preferredStyle: .alert)

        let identity = identities[indexPath.row]
        let removeAction = UIAlertAction(title: "Remove", style: .destructive) { [weak self] _ in
            guard let self = self else { return }
            let _ = self.sdk.remove(byIdentity: identity.identity, collectionName: SDKManager.collectionName)
            self.identities.removeAll(where: { $0.identity == identity.identity })
            tableView.reloadData()
        }

        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)

        alertController.addAction(removeAction)
        alertController.addAction(cancelAction)
        
        present(alertController, animated: true, completion: nil)
    }

}

