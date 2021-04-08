const DatasetDatabase = artifacts.require("DatasetDatabase.sol");

module.exports = function(deployer) {
    deployer.deploy(DatasetDatabase);
};
