const DatasetDatabase = artifacts.require("DatasetDatabase");
const truffleAssert = require('truffle-assertions');

contract("DatasetDatabase", accounts => {
    const mainAccount = accounts[0];
    const otherAccount = accounts[1];
    let instance = null;

    it('testing dataset registration and deletion', () => {
        let numberOfDatasets = 0;

        DatasetDatabase.deployed()
        .then(contractInstance => {
            instance = contractInstance;
            contractInstance.register_dataset('dataset hash', 'dataset description', 'dataset objective',
            {from: mainAccount})
        })
        .then(() => {
            return instance.datasets.call('dataset hash');
        })
        .then(dataset => {
            assert.equal(dataset['owner'], String(mainAccount), `The owner of this dataset should be ${String(mainAccount)}`);
            return instance.getNumberOfs.call();
        })
        .then(numberOfDatasets => {
            assert.equal(numberOfs, 1, 'Eactly 1 dataset should be registered to the contract.')
        })
        .then(() => {
            instance.delete_dataset('dataset hash');
        })
        .then(() => {
            numberOfDatasets = instance.getNumberOfs.call();
            return instance.models.call('dataset hash');
        })
        .then(dataset => {
            assert.isFalse(dataset['registered'], 'Dataset should have been deleted.');
            assert.equal(numberOfDatasets, 0, 'Dataset should have been deleted.')
        })
    })

    it('testing dataset registration error handling', () => {
        instance.register_dataset('dataset hash', 'dataset description', 'dataset objective',
                                {from: mainAccount})
        .then(() => {
            truffleAssert.reverts(instance.register_dataset('dataset hash', 'dataset description', 'dataset objective',
                                    {from: mainAccount}), 'Registration should be refused.');
        })
    })

    it('testing dataset modification functionality', () => {
        instance.register_dataset('dataset hash prime', 'dataset description', 'dataset objective',
                                {from: mainAccount})
        .then(() => {
            instance.modify_dataset('dataset hash prime', 'new description', 'new objective',
                                {from: mainAccount})
        })
        .then(() => {
            return instance.datasets.call('dataset hash prime')
        })
        .then(dataset => {
            assert.equal(dataset['description'], 'new description', 'Registered dataset failed to be modified.')
        })
        .then(() => {
            instance.delete_model('dataset hash prime');
        })
    })

    it('testing dataset modification security handling', () => {
        instance.register_dataset('dataset hash prime', 'dataset description', 'dataset objective',
                                {from: mainAccount})
        .then(() => {
            truffleAssert.reverts(instance.modify_dataset('dataset hash prime', 'new description', 'new objective',
                                {from: otherAccount}), 'Modifying this dataset should not be permitted');
        })
    })

})