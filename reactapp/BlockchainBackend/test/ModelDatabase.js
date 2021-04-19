const ModelDatabase = artifacts.require("ModelDatabase");
const truffleAssert = require('truffle-assertions');

contract("ModelDatabase", accounts => {
    const mainAccount = accounts[0];
    const otherAccount = accounts[1];
    let instance = null;

    it('testing model registration and deletion', () => {
        let numberOfModels = 0;

        ModelDatabase.deployed()
        .then(contractInstance => {
            instance = contractInstance;
            contractInstance.register_model('model hash', '', '', 'model description', 'model objective',
            {from: mainAccount})
        })
        .then(() => {
            return instance.models.call('model hash');
        })
        .then(model => {
            assert.equal(model['owner'], String(mainAccount), `The owner of this model should be ${String(mainAccount)}`);
            return instance.getNumberOfModels.call();
        })
        .then(numberOfModels => {
            assert.equal(numberOfModels, 1, 'Eactly 1 model should be registered to the contract.')
        })
        .then(() => {
            instance.delete_model('model hash');
        })
        .then(() => {
            numberOfModels = instance.getNumberOfModels.call();
            return instance.models.call('model hash');
        })
        .then(model => {
            assert.isFalse(model['registered'], 'Model should have been deleted.');
            assert.equal(numberOfModels, 0, 'Model should have been deleted.')
        })
    })

    it('testing model registration error handling', () => {
        instance.register_model('model hash',  '', '', 'model description', 'model objective',
                                {from: mainAccount})
        .then(() => {
            truffleAssert.reverts(instance.register_model('model hash', 'model description', 'model objective',
                                    {from: mainAccount}), 'Registration should be refused.');
        })
    })

    it('testing model modification functionality', () => {
        instance.register_model('model hash prime', '', '', 'model description', 'model objective',
                                {from: mainAccount})
        .then(() => {
            instance.modify_model('model hash prime', 'new description', 'new objective',
                                {from: mainAccount})
        })
        .then(() => {
            return instance.models.call('model hash prime')
        })
        .then(model => {
            assert.equal(model['description'], 'new description', 'Registered model failed to be modified.')
        })
        .then(() => {
            instance.delete_model('model hash prime');
        })
    })

    it('testing model modification security handling', () => {
        instance.register_model('model hash prime', '', '', 'model description', 'model objective',
                                {from: mainAccount})
        .then(() => {
            truffleAssert.reverts(instance.modify_model('model hash prime', 'new description', 'new objective',
                                {from: otherAccount}), 'Modifying this model should not be permitted');
        })
    })

})