pragma solidity >=0.5.16;

contract ModelDatabase {

    struct Model {
        address owner;
        string description;
        string objective;
        uint time;
        bool registered;
    }

    mapping(string => Model) public models;

    // function to register an Model
    function register_model(string memory _ipfsHash, string memory _description, string memory _objective) public {
        if (models[_ipfsHash].registered) {
            revert("This Model is already registered");
        }
        models[_ipfsHash] = Model({owner: msg.sender, description: _description, objective: _objective, time: block.timestamp, registered: true});
    }


    // Function to view information of certain Model
    function get_model_meta(string memory _ipfsHash) public view returns (address, string memory, string memory, uint) {
        return  (models[_ipfsHash].owner, models[_ipfsHash].description, models[_ipfsHash].objective, models[_ipfsHash].time);
    }

    // Allow owner to remove their Model from database
    function delete_model(string memory _ipfsHash) public {
        require(models[_ipfsHash].owner == msg.sender);
        delete(models[_ipfsHash]);
    }

}