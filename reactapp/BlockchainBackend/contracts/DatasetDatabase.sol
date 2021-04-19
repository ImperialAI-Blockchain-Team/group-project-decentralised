pragma solidity >=0.5.16;

contract Registry {
  function isDataOwner(address userAddress) public view returns(bool isIndeed) {}
}

contract DatasetDatabase {

    uint number;

    Registry registry;

    struct Dataset {
        address owner;
        string name;
        string description;
        string objective;
        uint time;
        bool registered;
    }

    mapping(string => Dataset) public datasets;

    string[] public hashes;

    mapping (string => bool) public names;
    string [] arrNames;

    constructor(address _contractAddressRegistry) public {
        registry = Registry(_contractAddressRegistry);
    }

    // function to register a Dataset
    function registerDataset(string memory _ipfsHash, string memory _name, string memory _description, string memory _objective) public {
        // Check is user is a registered data owner
        if (!registry.isDataOwner(msg.sender)){
            revert("Must be registered as a data owner");
        }

        if (datasets[_ipfsHash].registered) {
            revert("This Dataset is already registered");
        }

        // If name was used before, revert
        if (names[_name] == true){
            revert("Dataset name not unique");
        }

        datasets[_ipfsHash] = Dataset({owner: msg.sender,
                                       name: _name,
                                       description: _description,
                                       objective: _objective,
                                       time: block.timestamp,
                                       registered: true});

        hashes.push(_ipfsHash);

        // Mapping to ensure model name uniqueness
        names[_name] = true;
        arrNames.push(_name);
    }

    function getNumberOfs() public view returns(uint) {
        return hashes.length;
    }

    // function to modify dataset
    function modifyDataset(string memory _ipfsHash, string memory _description, string memory _objective) public {
        require(datasets[_ipfsHash].owner == msg.sender);
        datasets[_ipfsHash].description = _description;
        datasets[_ipfsHash].objective = _objective;
    }

    // Allow owner to remove their Dataset from database
    function deleteDataset(string memory _ipfsHash) public {
        require(datasets[_ipfsHash].owner == msg.sender);
        delete(datasets[_ipfsHash]);
    }

    // Get owner of particular dataset
    function getDatasetOwner(string memory _ipfsHash) public view returns(address) {
        return datasets[_ipfsHash].owner;
    }

    // Get name of dataset
    function getDatasetName(string memory _ipfsHash) public view returns(string memory) {
        return datasets[_ipfsHash].name;
    }

}
