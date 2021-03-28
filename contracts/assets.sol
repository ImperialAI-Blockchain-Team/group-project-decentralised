pragma solidity 0.8.3;

contract FederatedLearningAssets {

    struct Asset {
        uint id;
        address owner;
        address[] process;
        address[] download;
    }

    mapping(uint => Asset) public assets;
    uint public asset_counter;

    function register_asset(address[] _process, address[] _download) public {
        asset_counter ++;
        assets[asset_counter] = Asset({id: asset_counter, owner: msg.sender, process: _process, download: _download});
    }

    function modify_asset(uint _id, address[] _process, address[] _download) public {
        require(assets[_id].owner == msg.sender);
        assets[_id].process = _process;
        assets[_id].download = _download;
    }

    function transfer_ownership(uint _id, address _new_owner) public {
        require(assets[_id].owner == msg.sender);
        assets[_id].owner = _new_owner;
    }


}