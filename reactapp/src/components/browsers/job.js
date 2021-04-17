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
            targetJob: {}
            }
        this.handleOnKeyUp = this.handleOnKeyUp.bind(this);

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
        console.log(jobList)
        const renderedJobs = await jobList.map((job, jobID) => {
            console.log(job)
            console.log(jobID)
            return (
            <div className="jobContainer">
                <p><b>Owner</b>: {job['owner']}</p>
                <p><b>ID</b>: {jobID}</p>
                <p><b>Bounty</b>: {job['bounty']} wei </p>
                <p><b>Holding Fee</b>: {job['holdingFee']} wei </p>
                <p><b>Creation Date</b>: {new Date(job['initTime']*1000).toLocaleDateString()}</p>
                <p><b>Registration Deadline</b>: {new Date(job['initTime']*1000).toLocaleDateString()}</p>
                <p><button className="moreInfoButton" name={jobID} onClick={this.handleClick}>More Information</button>
                <Container2 triggerText={triggerText} job={jobID} />
                {/* <button id='like'>like</button> */}
                </p>
            </div>
            )
        })
        this.setState({renderedJobList: renderedJobs});
    }

    handleClick = async (event) => {
        const target = event.target;
        const id = target.name;
        console.log(id)
        const numRegistered = await jobsdatabase.methods.getNumRegistered(id).call();
        console.log(numRegistered)
        const registered = await jobsdatabase.methods.getJobRegistered(id).call();
        console.log(registered)

        let jobInfo = (
            <div className="jobInfo">
                <p><b>Info1</b>: something</p>
                <p><b>Info2</b>: something</p>
                <p><b>Info3</b>: something</p>
                <p><b>Info3</b>: something</p>
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