pragma solidity >=0.5.16;

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

    // function to register a Model
    function register_model(string memory _ipfsHash, string memory _description, string memory _objective) public {
        if (models[_ipfsHash].registered) {
            revert("This Model is already registered");
        }
        models[_ipfsHash] = Model({owner: msg.sender, description: _description, objective: _objective, time: block.timestamp, registered: true});
    }

    // function to modify model
    function modify_model(string memory _ipfsHash, string memory _description, string memory _objective) public {
        require(models[_ipfsHash].owner == msg.sender);
        models[_ipfsHash].description = _description;
        models[_ipfsHash].objective = _objective;
    }


    // Allow owner to remove their Model from database
    function delete_model(string memory _ipfsHash) public {
        require(models[_ipfsHash].owner == msg.sender);
        delete(models[_ipfsHash]);
    }

}