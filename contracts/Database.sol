pragma solidity 0.8.3;

contract AssetDatabase {

    struct Asset {
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
    }

}