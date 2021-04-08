pragma solidity >=0.5.16;

contract DatasetDatabase {

    uint number;

    struct Dataset {
        address owner;
        string description;
        string objective;
        uint time;
        bool registered;
    }

    mapping(string => Dataset) public datasets;
    string[] public hashes;

    // function to register a Dataset
    function register_model(string memory _ipfsHash, string memory _description, string memory _objective) public {
        if (datasets[_ipfsHash].registered) {
            revert("This Dataset is already registered");
        }
        datasets[_ipfsHash] = Dataset({owner: msg.sender, description: _description, objective: _objective, time: block.timestamp, registered: true});
        hashes.push(_ipfsHash);
    }

    function getNumberOfs() public view returns(uint) {
        return hashes.length;
    }

    // function to modify dataset
    function modify_dataset(string memory _ipfsHash, string memory _description, string memory _objective) public {
        require(datasets[_ipfsHash].owner == msg.sender);
        datasets[_ipfsHash].description = _description;
        datasets[_ipfsHash].objective = _objective;
    }

    // Allow owner to remove their Dataset from database
    function delete_dataset(string memory _ipfsHash) public {
        require(datasets[_ipfsHash].owner == msg.sender);
        delete(datasets[_ipfsHash]);
    }

}
