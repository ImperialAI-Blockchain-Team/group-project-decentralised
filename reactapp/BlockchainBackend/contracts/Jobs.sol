pragma solidity >=0.5.16;

contract ModelDatabase {

    function getModelOwner(string memory _ipfsHash) public view returns(address) {}
}


contract DatasetDatabase {

    function getDatasetOwner(string memory _ipfsHash) public view returns(address) {}
}

contract Registry {

  function isUser(address userAddress) public view returns(bool isIndeed) {}
}

contract Jobs {

    uint public jobsCreated = 0;

    DatasetDatabase datasetDatabase;
    ModelDatabase modelDatabase;
    Registry registry;

    struct Job {
        uint id;
        address owner;
        string modelIpfsHash;
        string[] registeredDatasetHashes;
        string strategyHash;
        uint minimalNbDatasets;
        uint initTime;
        uint daysUntilStart;
        bool active;
        bool registered;
    }

    mapping(uint => Job) public jobs;

    constructor(address _contractAddressRegistry, address _contractAddressDatasets, address _contractAddressModels) public {
        registry = Registry(_contractAddressRegistry);
        datasetDatabase = DatasetDatabase(_contractAddressDatasets);
        modelDatabase = ModelDatabase(_contractAddressModels);
    }

    function isSenderModelOwner(string memory _modelIpfsHash) public view returns(bool) {
        address owner = modelDatabase.getModelOwner(_modelIpfsHash);
        return (owner == msg.sender);
    }

    function isSenderDatasetOwner(string memory _datasetIpfsHash) public view returns(bool) {
        address owner = datasetDatabase.getDatasetOwner(_datasetIpfsHash);
        return (owner == msg.sender);
    }

    function getJobStartTime(uint _id) public view returns(uint) {
        return jobs[_id].initTime + jobs[_id].daysUntilStart * 1 days;
    }

    function registrationPeriodOver(uint _id) public view returns(bool) {
        return (block.timestamp >= getJobStartTime(_id));
    }

    function createJob(string memory _modelIpfsHash, string memory _strategyHash, uint _minClients, uint _daysUntilStart,
        uint _bounty, uint _holdingFee) public payable {

        // Check the Job creator is model owner
        require(isSenderModelOwner(_modelIpfsHash),"Only model owner can create a job for this model");

        // Check user has paid the correct bounty and job Creation Fee
        require(msg.value == jobCreationFee + _bounty,"Need to send the correct job creation fee and bounty");

        address payable[] memory init;

        jobs[jobsCreated] = Job({owner: msg.sender,
                                modelIpfsHash: _modelIpfsHash,
                                strategyHash: _strategyHash,
                                minimalNbDatasets: _minimalNbDatasets,
                                initTime: _initTime,
                                daysUntilStart: _daysUntilStart,
                                active: true,
                                registered: true}
                                );
    }


    function isSenderRegistered() public view returns(bool) {
        return registry.isUser(msg.sender);
    }

    function isSenderModelOwner(string memory _modelIpfsHash) public view returns(bool) {
        address owner = modelDatabase.getModelOwner(_modelIpfsHash);
        return (owner == msg.sender);
    }

    function isSenderDatasetOwner(string memory _modelIpfsHash) public view returns(bool) {
        address owner = datasetDatabase.getDatasetOwner(_modelIpfsHash);
        return (owner == msg.sender);
    }

    function getJobStartTime(uint jobId) public view returns(uint) {
        uint secondsUntilStart = jobs[jobId].daysUntilStart*24*60*60;
        return jobs[jobId].initTime + secondsUntilStart;
    }

    function getJobRegistered(uint _id) public view returns(address payable [] memory){
      return jobs[_id].arrDatasetOwners;
    }

    function getJobAllowed(uint _id) public view returns(address payable [] memory){
      return jobs[_id].arrAllowList;
    }

    function getJobDetails(uint _id) public view returns(address, uint, uint, uint){
        return (jobs[_id].owner, jobs[_id].minClients, jobs[_id].numAllow, jobs[_id].bounty);
    }

    function getJobStatus(uint _id) public view returns(uint, uint, bool, bool){
        return (jobs[_id].initTime, jobs[_id].daysUntilStart, jobs[_id].active, jobs[_id].trainingStarted);
    }


    function activateJob(uint jobId) public {
        require(jobs[jobId].owner == msg.sender);
        if (jobs[jobId].registered) {
            jobs[jobId].active = true;
        }
        else {
            revert("This job is not registered.");
        }
    }

    function deactivateJob(uint jobId) public {
        require(jobs[jobId].owner == msg.sender);
        // if registerDatasetOwner != []
        // functionality that registered hospitals
        // get their money back
        // if they were not payed for training yet
        if (jobs[jobId].registered) {
            jobs[jobId].active = false;
        }
        else {
            revert("This job is not registered.");
        }
    }
}
