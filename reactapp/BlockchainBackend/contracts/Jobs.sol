pragma solidity >=0.5.16;

contract ModelDatabase {
    function getModelOwner(string memory _ipfsHash) public view returns(address) {}
}

contract DatasetDatabase {
    function getDatasetOwner(string memory _ipfsHash) public view returns(address) {}
}

contract Jobs {
    
    uint public jobsCreated = 0;

    DatasetDatabase datasetDatabase;
    ModelDatabase modelDatabase;

    uint gracePeriod = 1 days;

    uint jobCreationFee = 1e6 wei;

    struct Job {
        address payable owner;
        string modelIpfsHash;
        string strategyHash;
        uint minClients;
        uint initTime;
        uint daysUntilStart;
        bool active;
        bool registered;
        bool trainingStarted;
        uint holdingFee;
        uint numAllow;
        uint bounty;
        uint feeSum;
        mapping(address => bool) datasetOwners;
        mapping(address => bool)  allowList;
        mapping(address => string) datasetHash;
        address payable[] arrDatasetOwners;
        address payable[] arrAllowList;
    }

    mapping(uint => Job) public jobs;

    constructor(address _contractAddressDatasets, address _contractAddressModels) public {
        datasetDatabase = DatasetDatabase(_contractAddressDatasets);
        modelDatabase = ModelDatabase(_contractAddressModels);
    }

    function isSenderModelOwner(string memory _modelIpfsHash) public view returns(bool) {
        address owner = modelDatabase.getModelOwner(_modelIpfsHash);
        return (owner == msg.sender);
    }

    function isSenderDatasetOwner(string memory _modelIpfsHash) public view returns(bool) {
        address owner = datasetDatabase.getDatasetOwner(_modelIpfsHash);
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
                                minClients: _minClients,
                                initTime: block.timestamp,
                                daysUntilStart: _daysUntilStart,
                                bounty: _bounty,
                                holdingFee: _holdingFee,
                                feeSum: 0,
                                numAllow: 0,
                                active: true,
                                registered: true,
                                trainingStarted: false,
                                arrDatasetOwners: init,
                                arrAllowList: init}
                                );

        jobsCreated = jobsCreated + 1;
    }


    // Called by dataset owners to register their willingness to join a jobs
    // Owners registers a particular dataset and
    function registerDatasetOwner(uint _id, string memory _datasetIpfsHash) public payable {
        // Check user is not model owner
        require(msg.sender != jobs[_id].owner, "Job owner cannot sign up as a client");

        // Check if user is dataset owner
        require(isSenderDatasetOwner(_datasetIpfsHash), "Need to be owner of this dataset");


        // Check job is still in registration period
        if (registrationPeriodOver(_id)){
            revert("Registration period for this job has ended");
        }

        // Check user has paid the correct holding fee
        require(msg.value == jobs[_id].holdingFee,"Need to send the correct holding fee");

        // Check job is active
        if (!jobs[_id].active){
            revert("This job is not currently active");
        }

        // Check if already registered
        if(jobs[_id].datasetOwners[msg.sender] == true){
            revert(" Already registered on this job");
        }

        // Add dataset owner's address to list of owner addresses comitted to job
        jobs[_id].datasetOwners[msg.sender] = true;
        jobs[_id].datasetHash[msg.sender] = _datasetIpfsHash;
        jobs[_id].arrDatasetOwners.push(msg.sender);

        // Sum up amount paid by committed dataowners
        jobs[_id].feeSum = jobs[_id].feeSum + msg.value;

    }
    
    // Called by Job owner to allow registered Clients to partake in job i.e. download model
    function addToAllowList(uint _id, address payable _datasetOwner) public {
        // Check user is job owner
        require(jobs[_id].owner == msg.sender, "Need to be registered as dataset owner");

        // Check address is in registry of interested owners
        require(jobs[_id].datasetOwners[_datasetOwner] == true, "Can only add an interested data owner to allow list");

        // Check job is still in registration period
        if (registrationPeriodOver(_id)){
            revert("Registration period for this job has ended");
        }

        // Check if dataset owner already registered on allowList
        if (jobs[_id].allowList[_datasetOwner] == true){
            revert("Dataset owner already added to allow list");
        }

        jobs[_id].allowList[_datasetOwner] = true;
        jobs[_id].arrAllowList.push(_datasetOwner);
    }

    function getNumRegistered(uint _id) public view returns(uint){
        return jobs[_id].arrDatasetOwners.length;
    }

    function getNumAllow(uint _id) public view returns(uint){
        return jobs[_id].arrAllowList.length;
    }

    function getJobDetails(uint _id) public view returns(address, uint, uint, uint){
        return (jobs[_id].owner, jobs[_id].minClients, jobs[_id].numAllow, jobs[_id].bounty);
    }

    function getJobStatus(uint _id) public view returns(uint, uint, bool, bool){
        return (jobs[_id].initTime, jobs[_id].daysUntilStart, jobs[_id].active, jobs[_id].trainingStarted);
    }

    function start_job(uint _id) public{
        // Can only be started by owner of job
        if (jobs[_id].owner != msg.sender){
            revert("Job can only be started by owner");
        }

        // Check registration period has ended
        if (!registrationPeriodOver(_id)){
            revert("Registration period for this job has not yet ended");
        }

        // Check grace period not over
        uint jobStartTime = getJobStartTime(_id);
        uint deadline = jobStartTime + gracePeriod;
        if (block.timestamp > deadline){
            revert("Too late to start training model");
        }

        if (jobs[_id].arrAllowList.length < jobs[_id].minClients) {
            revert("Minimum number of clients not reached");
        }

        // Set training started status as true
        jobs[_id].trainingStarted = true;

        // List of owners registered for this job
        address payable[] memory registeredDatasetOwners = jobs[_id].arrDatasetOwners;

        // Check every register data owner, if they are not on allow list, they get their fee back when training starts
        for (uint i=0; i<registeredDatasetOwners.length; i++) {
            if (jobs[_id].allowList[registeredDatasetOwners[i]] != true) {
                address payable payee = registeredDatasetOwners[i];
                payee.transfer(jobs[_id].holdingFee);
            }
        }

        // return job fee to client
        msg.sender.transfer(jobCreationFee);

    }

    function withdrawFee(uint _id) public{
        // Check grace period over
        uint jobStartTime = getJobStartTime(_id);
        uint deadline = jobStartTime + gracePeriod;
        if (block.timestamp < deadline){
            revert("Can only withdraw funds after training start deadline");
        }

        // To withdraw funds, need training not to have started
        require(!jobs[_id].trainingStarted, "Can only withdraw funds if training never started");

        // Only a data owner registered to job can withdraw funds
        require(jobs[_id].datasetOwners[msg.sender] == true, "Only an already registered data owner can withdraw fee");

        // Send holding fee to job registered user
        msg.sender.transfer(jobs[_id].holdingFee);

        // remove Client from datasetOwners list to prevent withdrawing fee again
        delete(jobs[_id].datasetOwners[msg.sender]);
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
