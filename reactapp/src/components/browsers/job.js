import React from "react";
import "./job.css"
import {Link} from 'react-router-dom';
import jobsdatabase from "../../contractInterfaces/jobsdatabase";
import modeldatabase from "../../contractInterfaces/modeldatabase";
import { Container2 } from "../helpers/Container2";
import web3 from "../../contractInterfaces/web3";
import ipfs from '../../ipfs'

export class JobBrowser extends React.Component {

    constructor(props) {

        super(props);
        this.state = {
            searchValue: '',
            ethAddress: '',
            numberOfJobs: -1,
            jobList: [],
            renderedJobList: (
                <div className="loadingCell">
                    <p><b> Loading ... </b></p>
                </div>
                ),
            jobInfo: this.browserIntroduction(),
            triggerText: "Register",
            targetJob: {},
            targetJobId: -1,
            targetJobDeadline: -1,
            targetJobGrace: -1,
            targetRegistered: [],
            targetAllowed: [],
            targetTrainingStarted: false

            }
        this.handleOnKeyUp = this.handleOnKeyUp.bind(this);
        this.startJob = this.startJob.bind(this);
        this.withdrawFeeClick = this.withdrawFeeClick.bind(this);
        this.downloadModel = this.downloadModel.bind(this);

        // call smart contract to render jobs
        this.getNumberOfJobs()
        .then(this.getJobList, (err) => {alert(err)})
        .then(this.renderJobs, (err) => {alert(err)});
    }

    browserIntroduction = () => {
        return (
            <div className="jobInfo">
                <h3>Click on a job to display additional info! </h3>
            </div>
        )
    };

    getNumberOfJobs = async () => {
        let numberOfJobs = await jobsdatabase.methods.jobsCreated().call();
        console.log(numberOfJobs)
        this.setState({numberOfJobs: numberOfJobs});
        return new Promise((resolve, reject) => {
            if (numberOfJobs != -1) {
                resolve(numberOfJobs);
            } else {
                reject(Error("Can't connect to JobsDatabase smart contract."))
            }
        })
    }

    getJobList = async (numberOfJobs) => {
        let newJobList = [];
        for (let i=0; i<numberOfJobs; i++) {
            const job = await jobsdatabase.methods.jobs(i).call();
            newJobList.push(job);
        }
        this.setState({jobList: newJobList})

        return new Promise((resolve, reject) => {
            resolve(newJobList);
        })
    }


    renderJobs = async (jobList) => {
        const { triggerText } = this.state.triggerText;
        const renderedJobs = await jobList.map((job, jobID) => {

            return (
            <div className="jobContainer">
                <p><b>Owner</b>: {job['owner']}</p>
                <p><b>ID</b>: {jobID}</p>
                <p><b>Bounty</b>: {job['bounty']} wei </p>
                <p><b>Holding Fee</b>: {job['holdingFee']} wei </p>
                <p><b>Creation Date</b>: {new Date(job['initTime']*1000).toUTCString()}</p>
                <p><b>Deadline</b>: {new Date((job['initTime'])*1000+parseInt(job['hoursUntilStart'])*60*60*1000).toUTCString()}</p>
                <p>
                    <button className="moreInfoButton" name={jobID} onClick={this.handleClick}>Job Details</button>
                    &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
                    <b>Register Dataset -></b> <Container2 triggerText={triggerText} job={jobID} />
                </p>
            </div>
            )
        })
        this.setState({renderedJobList: renderedJobs});
    }

    handleClick = async (event) => {
        const target = event.target;
        const id = target.name;
        const targetJob = await jobsdatabase.methods.jobs(id).call();
        const targetJobDeadline = targetJob['initTime'] + targetJob['hoursUntilStart']*60*60
        const targetJobGrace = targetJobDeadline + 1*24*60*60
        const targetTrainingStarted = targetJob['trainingStarted']


        const numAllowed = await jobsdatabase.methods.getNumAllow(id).call();
        const registered = await jobsdatabase.methods.getJobRegistered(id).call();
        const allowed = await jobsdatabase.methods.getJobAllowed(id).call();

        // Set target job info
        console.log(targetJob)
        this.setState({targetJob: targetJob})
        this.setState({targetJobId: id})
        this.setState({targetJobDeadline: targetJobDeadline})
        this.setState({targetJobGrace:targetJobGrace})
        this.setState({targetRegistered:registered})
        this.setState({targetAllowed:allowed})
        this.setState({targetTrainingStarted:targetTrainingStarted})

        // Get user to registered (committed) to job
        const registeredUsers = await registered.map((dataOwner, dataOwnerID) => {
            return (
                <p>
                    <b>Registered User {dataOwnerID+1}:</b> {dataOwner}
                    <button className="addAllowListButton" name={dataOwner} onClick={this.addClientAllow}>Add</button>
                </p>
            )
        })

        // Get allowed users i.e. registered data owners added to allow list by data-scientist
        const allowedUsers = await allowed.map((allowedUser, allowedUserID) => {
            return (
                <p>
                    <b>Allowed User {allowedUserID+1}:</b> {allowedUser}
                </p>
            )
        })

        this.setState({registeredUsers: registeredUsers});
        let jobInfo = (
            <div className="jobInfo">
                <h3> Registered Data Owners</h3>
                {registeredUsers}

                <h3> Job Owner-approved Clients</h3>
                {allowedUsers}

                <p>
                    <button className="startJobButton" name={this.state.targetJobId} onClick={this.startJob}>Start Training</button>
                &nbsp; &nbsp;
                    <button className="withdrawFundsButton" name={this.state.targetJobId} onClick={this.withdrawFeeClick}>Withdraw Fee</button>
                &nbsp; &nbsp;
                    <button className="downloadModelButton" name={this.state.targetJob} onClick={this.downloadModel}>Download Model</button>
                </p>
            </div>
            )
        this.setState({jobInfo: jobInfo})
    }

    handleOnKeyUp = async (event) => {
        const target = event.target;
        const name = target.name;
        await this.setState({searchValue: event.target.value});
        this.renderJobs(this.state.jobList);
    }

    addClientAllow = async (event) =>{
        const target = event.target;
        const clientAddress = target.name;

        const accounts = await web3.eth.getAccounts()

        // Only job owner can add clients to allow list
        let isJobOwner = this.state.targetJob["owner"] == accounts[0]
        if (!isJobOwner){
            alert("Not job owner, only job owner can add registered users to job allow list")
            return;
        }

        // Check is registration period is over
        let isRegistrationOver = this.state.targetJobDeadline > (Date.now()/1000)
        if (isRegistrationOver){
            alert("Registration period over, can't add more clients.")
            return;
        }

        // Check is user has already been added to allow list
        let alreadyAllowed = this.state.targetAllowed.includes(clientAddress)
        if (alreadyAllowed){
            alert("Cannot add a data owner twice, this data owner has already been added to the allow list")
            return;
        }

        // add registered owner to allow list
        //arg: _jobID, _datasetOwner
        await jobsdatabase.methods.addToAllowList(this.state.targetJobId, clientAddress).send({from: accounts[0]})
        .on('transactionHash', (hash) =>{
            console.log(hash);
        })
        .on('error', async (error, receipt) => {
            console.log(error);
            if (receipt) {
               console.log(receipt);
            }
        })

    }

    startJob = async (event) =>{
        const target = event.target;
        const id = target.name;
        console.log(id);

        const accounts = await web3.eth.getAccounts();

        // Check if user is job owner
        let isJobOwner = this.state.targetJob["owner"] == accounts[0]
        if (!isJobOwner){
            alert("Not job owner, only job owner can start training")
            return;
        }

        // Check registration period is over
        let isRegistrationOver = this.state.targetJobDeadline > (Date.now()/1000)
        if (!isRegistrationOver){
            alert("Registration Period not over, can only start period after registration deadline")
            return;
        }

        // Check if minimum clients available to start job
        let minClients = this.state.targetJob["minClients"]
        let isMinClients = this.state.targetAllowed.length >= minClients
        if (!isMinClients){
            alert("Not enough clients to start job")
            return;
        }

        // TODO
        // Change to camelCase
        // start training for job
        //arg: _jobID, _datasetOwner
        await jobsdatabase.methods.start_job(this.state.targetJobId).send({from: accounts[0]})
        .on('transactionHash', (hash) =>{
            console.log(hash);
        })
        .on('error', async (error, receipt) => {
            console.log(error);
            if (receipt) {
               console.log(receipt);
            }
        })

    }

    withdrawFeeClick = async (event) =>{
        const target = event.target;
        const id = target.name;
        console.log(id);

        const accounts = await web3.eth.getAccounts();

        let isGraceOver = (Date.now()/1000) > this.state.targetJobGrace;
        if(!isGraceOver){
            alert("Cannot withdraw funds yet")
            return;
        }

        let hasTrained = this.state.targetJob["trainingStarted"];
        if(hasTrained){
            alert("Cannot withdraw funds for this job")
            return;
        }

        // Check is user has already been registered
        let alreadyRegistered = this.state.registered.includes(accounts[0])
        if (alreadyRegistered){
            alert("Cannot register twice, you have already registered to this job")
            return;
        }

        await jobsdatabase.methods.withdrawFee(id).send({from: accounts[0]})
        .on('transactionHash', (hash) =>{
            console.log(hash);
        })
        .on('error', async (error, receipt) => {
            console.log(error);
            if (receipt) {
               console.log(receipt);
            }
        })

    }

    downloadModel = async () => {
        const accounts = await web3.eth.getAccounts();

        // Check is user has already been added to allow list
        let isAllowed = this.state.targetAllowed.includes(accounts[0])
        if (!isAllowed){
            alert("Only data owners on the job's allow list can download the model")
            return;
        }

        // Check if training period
        let isTrainingStarted = this.state.targetTrainingStarted
        if (!isTrainingStarted){
            alert("Can only download model during training period")
            return
        }

        const cid = this.state.targetJob["modelIpfsHash"]
        console.log(cid)
        const chunks = await ipfs.cat(cid)

        console.log(chunks.toString())

        const element = document.createElement("a");
        const file = new Blob([chunks], {type: 'uint8'});
        element.href = URL.createObjectURL(file);
        element.download = "model.py";
        element.click();

    }

    render() {

        return (
            <div className="pageContainer">
                <div className="headerContainer">
                    <div className="searchBarContainer">

                        <input type="text" id="myInput" onKeyUp={this.handleOnKeyUp} placeholder="Search model (by description)" />
                    </div>
                    <p id="numberOfJobs">{this.state.numberOfJobs} jobs already uploaded to the system</p>
                    <Link to="/clientPackage.zip" target="_blank" download>Client Package</Link>
                    <hr />
                </div>
                <div className="resultContainer">
                    <tr>
                        {this.state.renderedJobList}
                    </tr>
                </div>
                <div className="jobInfoContainer">
                    {this.state.jobInfo}
                </div>
            </div>
        )
    }
}