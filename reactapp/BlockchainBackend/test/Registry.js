var Registry = artifacts.require("./Registry.sol");

contract("Registry", function (accounts) {
    var instance = null; // store the Users contract instance
    var mainAccount = accounts[0];
    var anotherAccount = accounts[1];

    // testing registering a new user and checking if the total users has increased and if the

    it("should register a user", function() {
        var usersBeforeRegister = null;

        return Registry.deployed().then(function(contractInstance) {
            instance = contractInstance; // storing the contract instance
            return instance.getUserCount.call(); // calling the smart contract function getUserCount to get the current number of users
        }).then(function(result) {
            usersBeforeRegister = result.toNumber(); // storing the current number on the var usersBeforeRegister
            return instance.insertUser('Test User', 'true','true','true', {
                from: mainAccount
            }); // registering the user calling the smart contract function registerUser
        }).then(function(result) {
            return instance.getUserCount.call();
        }).then(function(result) {
            // checking if the total number of user is increased by 1
            assert.equal(result.toNumber(), (usersBeforeRegister+1), "number of users must be (" + usersBeforeRegister + " + 1)");
            // calling the smart contract function isRegistered to know if the sender is registered.
            return instance.isUser(mainAccount);
        }).then(function(result) {
            // we are expecting a boolean in return that it should be TRUE
            assert.isTrue(result);
        });
    }); // end of "should register an user"user

    // Tests if data on blockchain matches with the data given during the registration.

    it("username and status should be the same the one gave on the registration", function() {
        return instance.getUser(mainAccount).then(function(result) {
            assert.equal(result[0], 'Test User');
            assert.equal(result[1], true);
            assert.equal(result[2], true);
            assert.equal(result[3], true);
        });
    }); // end testing username and status

    it("update profile", function() {
        return instance.updateUserType(mainAccount, false,false,false, {
            from: mainAccount
        }).then(function(result) {
            return instance.getUser(mainAccount);
        }).then(function(result) {
            assert.equal(result[0], 'Test User');
            assert.equal(result[1], false);
            assert.equal(result[2], false);
            assert.equal(result[3], false);
        });
    }); // end should update the profile

    it("not a user", function(){
        return instance.isUser(anotherAccount)
        .then(bool => {
            assert.equal(bool, false, "This account should not be registered.")
        })
    })

    it("have 2 users", function() {

        return Registry.deployed().then(function(contractInstance) {
            //instance = contractInstance; // storing the contract instance
            return instance.getUserCount.call(); // calling the smart contract function getUserCount to get the current number of users
        }).then(function(result) {
            usersBeforeRegister = result.toNumber(); // storing the current number on the var usersBeforeRegister
            return instance.insertUser('Test User 2', 'true','true','true', {
                from: anotherAccount
            }); // registering the user calling the smart contract function registerUser
        }).then(function(result) {
            return instance.getUserCount.call();
        }).then(function(result) {
            // checking if the total number of user is increased by 1
            assert.equal(result.toNumber(), (usersBeforeRegister+1), "number of users must be (" + usersBeforeRegister + " + 1)");
            // calling the smart contract function isRegistered to know if the sender is registered.
            return instance.isUser(anotherAccount);
        }).then(function(result) {
            // we are expecting a boolean in return that it should be TRUE
            assert.isTrue(result);
        });
    }); // end of "should register an user"user

});
