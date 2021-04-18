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
    
    uint private jobsCreated = 0;
    
    Registry registry;
    DatasetDatabase datasetDatabase;
    ModelDatabase modelDatabase;

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

    function createJob(string memory _modelIpfsHash, string memory _strategyHash, uint _minimalNbDatasets, uint _daysUntilStart) public {
        require(isSenderModelOwner(_modelIpfsHash));
        jobsCreated = jobsCreated + 1;
        uint _initTime = block.timestamp;
        string[] memory _registeredDatasetHashes;
        jobs[jobsCreated] = Job( 
                                {id: jobsCreated, 
                                owner: msg.sender,
                                modelIpfsHash: _modelIpfsHash, 
                                registeredDatasetHashes: _registeredDatasetHashes, 
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
    
    function registrationPeriodOver(uint jobId) public view returns(bool) {
        return (block.timestamp >= getJobStartTime(jobId));
    }
    
    /*function registerDatasetOwner(uint jobId) public {
        string memory _modelIpfsHash = jobs[jobId].modelIpfsHash;
        require(isSenderDatasetOwner(_modelIpfsHash));
        // transaction needed
        // specific dataset hash
        // time not over
        // active
    }*/
    
    function getJobDetails() public view {
        require(isSenderRegistered());
    //   only possible for registered users
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
