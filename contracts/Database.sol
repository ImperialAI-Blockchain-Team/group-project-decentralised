pragma solidity 0.8.3;

contract AssetDatabase {

    struct Asset {
        address owner;
        string description;
        string objective;
        uint time;
    }

    mapping(string => Asset) public assets;

    // function to register an asset
    function register_asset(string memory _ipfsName, string memory _description, string memory _ipfsObjective) public {
        assets[_ipfsName] = Asset({owner: msg.sender, description: _description, objective: _ipfsObjective, time: block.timestamp});
    }


    // Function to view information of certain asset
    function get_asset(string memory _ipfsName) public view returns (address, string memory, string memory, uint) {


        return  (assets[_ipfsName].owner, assets[_ipfsName].description, assets[_ipfsName].objective, assets[_ipfsName].time);
    }

    // Allow owner to remove their asset from database
    function delete_asset(string memory _ipfsName) public {
        require(assets[_ipfsName].owner == msg.sender);
        delete(assets[_ipfsName]);
    }
    
}