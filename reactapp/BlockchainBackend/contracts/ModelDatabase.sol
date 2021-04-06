pragma solidity >=0.5.16;

contract ModelDatabase {

    struct Model {

        address owner;
        string description;
        string objective;
        uint time;
        bool registered;
    }

    mapping(string => Asset) public assets;

    // function to register an asset
    function register_asset(string memory _ipfsName, string memory _description, string memory _ipfsObjective) public {
        // Check if asset is already registered to prevent overwriting existing entry
        if (assets[_ipfsName].registered) {
            revert("This asset is already registered");
        }
        // Register new asset
        assets[_ipfsName] = Asset({owner: msg.sender, description: _description, objective: _ipfsObjective, time: block.timestamp, registered: true});
    }


    // Function to view information of certain asset
    function get_asset_meta(string memory _ipfsName) public view returns (address, string memory, string memory, uint) {

        return  (assets[_ipfsName].owner, assets[_ipfsName].description, assets[_ipfsName].objective, assets[_ipfsName].time);
    }

    // Allow owner to remove their asset from database
    function delete_asset(string memory _ipfsName) public {
        require(assets[_ipfsName].owner == msg.sender);
        delete(assets[_ipfsName]);

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