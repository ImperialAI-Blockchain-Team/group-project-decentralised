const ModelDatabase = artifacts.require("ModelDatabase.sol");

const fs = require('fs');

let contract = JSON.parse(fs.readFileSync('./../build/contracts/Registry.json', 'utf8'));
const contractAddressRegistry = contract.networks['3']['address'];


module.exports = function(deployer) {
 deployer.deploy(ModelDatabase, contractAddressRegistry);
};