pragma solidity >=0.5.16;

contract Registry {
  function isDataScientist(address userAddress) public view returns(bool isIndeed) {}
}

contract ModelDatabase {

    uint number;

    struct Model {
        address owner;
        string description;
        string objective;
        uint time;
        bool registered;
    }

    mapping(string => Model) public models;
    string[] public hashes;

    Registry registry;

    constructor(address _contractAddressRegistry) public {
        registry = Registry(_contractAddressRegistry);
    }

    // function to register a Model
    function registerModel(string memory _ipfsHash, string memory _description, string memory _objective) public {
        if (!registry.isDataScientist(msg.sender)){
            revert("Must be registered as a data scientist");
        }

        if (models[_ipfsHash].registered) {
            revert("This Model is already registered");
        }
        models[_ipfsHash] = Model({owner: msg.sender, description: _description, objective: _objective, time: block.timestamp, registered: true});
        hashes.push(_ipfsHash);
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

}