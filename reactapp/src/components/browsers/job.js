import React from "react";
import "./job.css"
import {Link} from 'react-router-dom';
import jobsdatabase from "../../contractInterfaces/jobsdatabase";
import modeldatabase from "../../contractInterfaces/modeldatabase";
import FormDialog from "../forms/jobSignup"
import { Container2 } from "../helpers/Container2";
import {Container} from "../helpers/Container";

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
            targetJobId: -1

            }
        this.handleOnKeyUp = this.handleOnKeyUp.bind(this);
        this.startJob = this.startJob.bind(this);
        this.withdrawFee = this.withdrawFee.bind(this);

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
            console.log(parseInt(job['daysUntilStart'])*24*60*60)
            console.log(jobID)
            return (
            <div className="jobContainer">
                <p><b>Owner</b>: {job['owner']}</p>
                <p><b>ID</b>: {jobID}</p>
                <p><b>Bounty</b>: {job['bounty']} wei </p>
                <p><b>Holding Fee</b>: {job['holdingFee']} wei </p>
                <p><b>Creation Date</b>: {new Date(job['initTime']*1000).toLocaleDateString()}</p>
                <p><b>Registration Deadline</b>: {new Date((job['initTime'])*1000+parseInt(job['daysUntilStart'])*24*60*60*1000).toLocaleDateString()}</p>
                <p>
                    <button className="moreInfoButton" name={jobID} onClick={this.handleClick}>Job Details</button>
                    &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
                <Container2 triggerText={triggerText} job={jobID} />
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
        console.log(targetJob);
        const numAllowed = await jobsdatabase.methods.getNumRegistered(id).call();
        const registered = await jobsdatabase.methods.getJobRegistered(id).call();
        const allowed = await jobsdatabase.methods.getJobAllowed(id).call();

        this.setState({targetJob: targetJob})
        this.setState({targetJobId: id})

        const registeredUsers = await registered.map((dataOwner, dataOwnerID) => {
            return (
                <p><b>Registered User {dataOwnerID}:</b> {dataOwner}</p>
            )
        })

        this.setState({registeredUsers: registeredUsers});
        let jobInfo = (
            <div className="jobInfo">
                <h3> Registered Data Owners</h3>
                {registeredUsers}

                <h3> Job Owner-approved Clients</h3>

                <p>
                    <button className="startJobButton" name={this.state.targetJobId} onClick={this.startJob}>Start Training</button>
                &nbsp; &nbsp; &nbsp;
                    <button className="withdrawFundsButton" name={this.state.targetJobId} onClick={this.withdrawFee}>Withdraw Fee</button>
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

    startJob = async (event) =>{
        const target = event.target;
        const id = target.name;
        console.log(id);

    }

    withdrawFee = async (event) =>{
        const target = event.target;
        const id = target.name;
        console.log(id);

    }

    render() {

        return (
            <div className="pageContainer">
                <div className="headerContainer">
                    <div className="searchBarContainer">

                        <input type="text" id="myInput" onKeyUp={this.handleOnKeyUp} placeholder="Search model (by description)" />
                    </div>
                    <p id="numberOfJobs">{this.state.numberOfJobs} jobs already uploaded to the system</p>
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