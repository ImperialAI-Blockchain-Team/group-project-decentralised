const Jobs = artifacts.require("Jobs.sol");

const fs = require('fs');

let contract1 = JSON.parse(fs.readFileSync('./../build/contracts/DatasetDatabase.json', 'utf8'));
const contractAddressDatasets = contract1.networks['3']['address'];

let contract2 = JSON.parse(fs.readFileSync('./../build/contracts/ModelDatabase.json', 'utf8'));
const contractAddressModels = contract2.networks['3']['address'];


module.exports = function(deployer) {
 deployer.deploy(Jobs, contractAddressDatasets, contractAddressModels);
};