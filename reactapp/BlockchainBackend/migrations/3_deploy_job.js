const Jobs = artifacts.require("Jobs.sol");
const fs = require('fs');

var contract = JSON.parse(fs.readFileSync('./../build/contracts/Registry.json', 'utf8'));
var contractAddressRegistry = contract.networks['3']['address'];

var contract = JSON.parse(fs.readFileSync('./../build/contracts/DatasetDatabase.json', 'utf8'));
var contractAddressDatasets = contract.networks['3']['address'];

var contract = JSON.parse(fs.readFileSync('./../build/contracts/ModelDatabase.json', 'utf8'));
var contractAddressModels = contract.networks['3']['address'];

module.exports = function(deployer) {
 deployer.deploy(Jobs, contractAddressRegistry, contractAddressDatasets, contractAddressModels);
};
