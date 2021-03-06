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

    uint public gracePeriod = 1 hours;

    uint public jobCreationFee = 1e6 wei;

    uint public holdingFee = 1e6 wei;

    address payable private serverAddress = 0xc6c3F7d3E6907ea2C3174c06d188Eb57c59BF59c;

    struct Job {
        address payable owner;
        string modelIpfsHash;
        string strategyHash;
        string testDatasetHash;
        uint minClients;
        uint initTime;
        uint hoursUntilStart;
        bool active;
        bool trainingStarted;
        bool trainingEnded;
        uint bounty;
        mapping(address => bool) datasetOwners;
        mapping(address => bool)  allowList;
        mapping(address => string) datasetHash;
        address payable[] arrDatasetOwners;
        address payable[] arrAllowList;
    }

    struct JobResults {
        string resultsHash;
        string weightsHash;
        mapping(address => uint) compensation;
        mapping(address => bool) isCompensated;
    }

    mapping(uint => Job) public jobs;
    mapping(uint => JobResults) public jobResults;

    constructor(address _contractAddressDatasets, address _contractAddressModels) public {
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
        return jobs[_id].initTime + jobs[_id].hoursUntilStart * 1 hours;
    }

    function getGraceDeadline(uint _id) public view returns(uint){
        return getJobStartTime(_id) + gracePeriod;
    }

    function registrationPeriodOver(uint _id) public view returns(bool) {
        return (block.timestamp >= getJobStartTime(_id));
    }

    function createJob(string memory _modelIpfsHash, string memory _strategyHash, string memory _testDatasetHash, uint _minClients, uint _hoursUntilStart,
        uint _bounty) public payable {

        // Check the Job creator is model owner
        require(isSenderModelOwner(_modelIpfsHash),"Only model owner can create a job for this model");

        // Check user has paid the correct bounty and job Creation Fee
        require(msg.value == jobCreationFee + _bounty,"Need to send the correct job creation fee and bounty");

        address payable[] memory init;

        jobs[jobsCreated] = Job({owner: msg.sender,
                                modelIpfsHash: _modelIpfsHash,
                                strategyHash: _strategyHash,
                                testDatasetHash: _testDatasetHash,
                                minClients: _minClients,
                                initTime: block.timestamp,
                                hoursUntilStart: _hoursUntilStart,
                                bounty: _bounty,
                                active: true,
                                trainingStarted: false,
                                trainingEnded: false,
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
        require(msg.value == holdingFee,"Need to send the correct holding fee");


        // Check if already registered
        if(jobs[_id].datasetOwners[msg.sender] == true){
            revert(" Already registered on this job");
        }

        // Add dataset owner's address to list of owner addresses comitted to job
        jobs[_id].datasetOwners[msg.sender] = true;
        jobs[_id].datasetHash[msg.sender] = _datasetIpfsHash;
        jobs[_id].arrDatasetOwners.push(msg.sender);

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

    function getJobRegistered(uint _id) public view returns(address payable [] memory){
      return jobs[_id].arrDatasetOwners;
    }

    function getJobAllowed(uint _id) public view returns(address payable [] memory){
      return jobs[_id].arrAllowList;
    }

    function getJobDetails(uint _id) public view returns(address, uint, uint){
        return (jobs[_id].owner, jobs[_id].minClients, jobs[_id].bounty);
    }

    function getJobStatus(uint _id) public view returns(uint, uint, bool, bool){
        return (jobs[_id].initTime, jobs[_id].hoursUntilStart, jobs[_id].trainingStarted, jobs[_id].trainingEnded);
    }

    function getMinClients(uint _id) public view returns(uint) {
        return jobs[_id].minClients;
    }

    function startJob(uint _id) public{
        // Can only be started by owner of job
        if (jobs[_id].owner != msg.sender){
            revert("Job can only be started by owner");
        }

        // Check registration period has ended
        if (!registrationPeriodOver(_id)){
            revert("Registration period for this job has not yet ended");
        }

        // Check grace period not over
        //uint jobStartTime = getJobStartTime(_id);
        uint deadline = getGraceDeadline(_id);
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

        // return job fee to client
        msg.sender.transfer(jobCreationFee);
        //serverAddress.transfer(jobCreationFee);

    }


    function withdrawFee(uint _id) public{

        // Only a data owner registered to job can withdraw funds
        require(jobs[_id].datasetOwners[msg.sender] == true, "Only an already registered data owner can withdraw fee");

        // Check grace period over
        uint jobStartTime = getJobStartTime(_id);
        uint deadline = getGraceDeadline(_id);
        if (block.timestamp < jobStartTime ){
            revert("Can only withdraw funds after registration period end");
        }

        // Allow registered users not on the allow list to withdraw fee after registration deadline
        if (jobs[_id].allowList[msg.sender] != true) {
                msg.sender.transfer(holdingFee);
                // remove Client from datasetOwners list to prevent withdrawing fee again
                delete(jobs[_id].datasetOwners[msg.sender]);
        }

        // If not a failed job i.e. min clients available, then can only withdraw funds if job owner does not start training
        // by the grace period deadline
        if ((block.timestamp < deadline) && (jobs[_id].arrAllowList.length >= jobs[_id].minClients)){
            revert("Can only withdraw funds after training start deadline");
        }

        // To withdraw funds for rest of clients i.e. ones on allow list, need training not to have started
        require(!jobs[_id].trainingStarted, "Can only withdraw funds if training never started");

        // Send holding fee to job registered user
        msg.sender.transfer(holdingFee);

        // remove Client from datasetOwners list to prevent withdrawing fee again
        delete(jobs[_id].datasetOwners[msg.sender]);
    }

    function endFailedJob(uint _id) public {
        // Only the owner of the job can end it
        require(jobs[_id].owner == msg.sender, "Only job owner can end job");

        // Check grace period over
        uint jobStartTime = getJobStartTime(_id);
        if (block.timestamp < jobStartTime){
            revert("Job can only end without training after registration ends");
        }
        
        // To end job, training should not have started
        require(!jobs[_id].trainingStarted, "Can only withdraw funds if training never started");
        
        // To end job, length of the allow list must be smaller than minClients
        require(getNumAllow(_id) < getMinClients(_id), "Sufficient number of clients on allow list to start training");

        // Can only end an active job, to prevent spamming function
        require(jobs[_id].active == true);

        // withdraw bounty to data scientist
        msg.sender.transfer(jobs[_id].bounty);
        msg.sender.transfer(jobCreationFee);

        jobs[_id].active = false;
    }

    function isRegistered(uint _id) public view returns(bool){
        return jobs[_id].datasetOwners[msg.sender];
    }

    function compensate(uint _id, uint [] memory _compensation, address payable [] memory _clients,
                        string memory _resultsHash, string memory _weightsHash) public{

        require(msg.sender == serverAddress,"Only server can end training and initiate compensation");

        require(_compensation.length == _clients.length,"Number of clients must match compensation amounts");

        require(_clients.length == jobs[_id].arrAllowList.length, "Number of clients must match number of allowed users");

        if(jobs[_id].trainingEnded == true){
            revert("Training has ended and compensation has already been spent out to participants");
        }

        uint compensationSum = 0;
        for (uint j=0; j<_compensation.length; j++) {
            compensationSum = compensationSum + _compensation[j];
        }

        // Check sum of compensation array is less than job bounty
        require(compensationSum <= jobs[_id].bounty, "Can't pay out more than bounty amount");

        // Assign compensation amount to each client
        for (uint i=0; i<_clients.length; i++) {
            address payable payee = _clients[i];
            jobResults[_id].compensation[payee] = _compensation[i];
        }

        // Set status of job as ended
        jobs[_id].trainingEnded = true;
        jobs[_id].active = false;
        // Log results of job
        jobResults[_id].resultsHash = _resultsHash;
        jobResults[_id].weightsHash = _weightsHash;
    }

    function getCompensation(uint _id) public{

        // Check if training for this job has successfully ended
        require(jobs[_id].trainingEnded == true, "Can't receive compensation until successful training");

        // Check if user was on the Allow list of the job
        require(jobs[_id].allowList[msg.sender] == true, "Can't receive compensation if not a job participant");

        // Ensure user is not compensated twice, can only receive compensation for job if not compensated before
        if (jobResults[_id].isCompensated[msg.sender] == true){
            revert("Already received compensation for this job");
        }

        // Transfer to user their compensation amount
        msg.sender.transfer(jobResults[_id].compensation[msg.sender]);
        msg.sender.transfer(holdingFee);

        // Log that user has been compensated
        jobResults[_id].isCompensated[msg.sender] = true;
    }

    function getWeights(uint _id) public view returns(string memory){
        // Check user is job owner
        require(msg.sender == jobs[_id].owner,"Only owner can access weights");

        return(jobResults[_id].weightsHash);
    }

    function getCompensationResults(uint _id) public view returns(string memory){
        // Check user is a job participant
        require((msg.sender == jobs[_id].owner) || (jobs[_id].allowList[msg.sender] == true) ,"Only a job participant can access results");

        return(jobResults[_id].resultsHash);
    }

    function isCompensated(uint _id) public view returns(bool){
        // Check if user is a job participant first
        require(jobs[_id].allowList[msg.sender]==true,"User not eligible for compensation for this job");

        //Check if user has been isCompensated
        return(jobResults[_id].isCompensated[msg.sender]);
    }

    function getDataset(uint _id, address datasetOwner) public view returns(string memory){
        return jobs[_id].datasetHash[datasetOwner];
    }

}