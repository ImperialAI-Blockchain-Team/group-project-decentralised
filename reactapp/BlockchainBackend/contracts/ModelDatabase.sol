pragma solidity >=0.5.16;

contract Registry {
  function isDataScientist(address userAddress) public view returns(bool isIndeed) {}
  function isDataOwner(address userAddress) public view returns(bool isIndeed) {}
}

contract ModelDatabase {

    uint number;

    struct Model {
        address owner;
        string name;
        string objective;
        string description;
        string dataRequirements;
        uint interest;
        mapping(address => bool) datasetOwnersInterest;
        address [] arrDatasetOwnersInterest;
        uint time;
        bool registered;
    }

    mapping(string => Model) public models;
    string[] public hashes;
    mapping (string => bool) public names;
    string [] arrNames;

    Registry registry;

    constructor(address _contractAddressRegistry) public {
        registry = Registry(_contractAddressRegistry);
    }

    // function to register a Model
    function registerModel(string memory _ipfsHash,
                            string memory _name,
                            string memory _objective,
                            string memory _description,
                            string memory _dataRequirements) public {
        if (!registry.isDataScientist(msg.sender)){
            revert("Must be registered as a data scientist");
        }

        if (models[_ipfsHash].registered) {
            revert("This Model is already registered");
        }

        // If name was used before, revert
        if (names[_name] == true){
            revert("Model name not unique");
        }

        address [] memory init;

        models[_ipfsHash] = Model({owner: msg.sender,
                                name: _name,
                                description: _description,
                                objective: _objective,
                                dataRequirements: _dataRequirements,
                                interest: 0,
                                arrDatasetOwnersInterest: init,
                                time: block.timestamp,
                                registered: true});
        hashes.push(_ipfsHash);

        // Mapping to ensure model name uniqueness
        names[_name] = true;
        arrNames.push(_name);
    }

    function getNumberOfModels() public view returns(uint) {
        return hashes.length;
    }

    // function to modify model
    function modifyModel(string memory _ipfsHash, string memory _description, string memory _objective) public {
        require(models[_ipfsHash].owner == msg.sender);
        models[_ipfsHash].description = _description;
        models[_ipfsHash].objective = _objective;
    }


    // Allow owner to remove their Model from database
    function deleteModel(string memory _ipfsHash) public {
        require(models[_ipfsHash].owner == msg.sender);
        delete(models[_ipfsHash]);
    }

    function getModelOwner(string memory _ipfsHash) public view returns(address) {
        return models[_ipfsHash].owner;
    }

    function getModelName(string memory _ipfsHash) public view returns(string memory){
        return models[_ipfsHash].name;
    }

    // Allow Data owners to register interest in a model
    function registerInterest(string memory _ipfsHash) public {
        // Ensure only data owners can register interest
        if (!registry.isDataOwner(msg.sender)){
            revert("Only data owners can register interest");
        }
        // Ensures data-owners can register interest in model only once
        if (models[_ipfsHash].datasetOwnersInterest[msg.sender] == true){
            revert("User already registered interest");
        }

        models[_ipfsHash].interest = models[_ipfsHash].interest + 1;
        models[_ipfsHash].datasetOwnersInterest[msg.sender] = true;
        models[_ipfsHash].arrDatasetOwnersInterest.push(msg.sender);
    }

    // Get array of interested users
    function getModelInterested(string memory _ipfsHash) public view returns(address [] memory){
      return models[_ipfsHash].arrDatasetOwnersInterest;
    }

}