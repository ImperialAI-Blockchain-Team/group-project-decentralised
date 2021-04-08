const ModelDatabase = artifacts.require("ModelDatabase.sol");
const DatasetDatabase = artifacts.require("DatasetDatabase.sol");
const Registry = artifacts.require("Registry.sol");

module.exports = function(deployer) {
 deployer.deploy(Registry)
 deployer.deploy(ModelDatabase);
 deployer.deploy(DatasetDatabase);
};