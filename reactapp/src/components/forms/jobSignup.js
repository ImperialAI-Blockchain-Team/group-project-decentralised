import React from "react";
import "./job.css";
import jobsdatabase from '../../contractInterfaces/jobsdatabase'
import web3 from "../../contractInterfaces/web3";

function validate(modelName, description, buffer){
    // Validate inputs, can add more detailed errors afterwards
    const errors = [];

    if (modelName.length === 0) {
        errors.push("Name can't be empty");
    }
    if (description.length === 0){
        errors.push("Description can't be empty")
    }
    if (buffer.length === 0){
        errors.push("Have to upload model")
    }
    return errors
}

export class JobSignup extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            ipfsHash:'',
            transactionHash: '',
            txReceipt: '',
            formErrors: []
        };
        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }

    handleChange(event) {
        const target = event.target;
        const name = target.name;
        this.setState({[name]: event.target.value});
    }

    handleSubmit = async (event) => {
        event.preventDefault()
        //bring in user's metamask account address
        const accounts = await web3.eth.getAccounts();

        // Get Job data
        const targetJob = await jobsdatabase.methods.jobs(this.props.job).call()
        const registered = await jobsdatabase.methods.getJobRegistered(this.props.job).call()

        // First check if user is registered
        let isDataOwner = await jobsdatabase.methods.isSenderDatasetOwner(this.state.ipfsHash).call({from : accounts[0]});
        console.log(isDataOwner )
        if (!isDataOwner ){
            alert("Not valid dataset or not dataset owner, only dataset owner can register for this job")
            return;
        }

        // Check if registration period is not over
        const registrationDeadline = +targetJob['initTime'] + targetJob['hoursUntilStart']*60*60
        let isRegistrationOver =  registrationDeadline > (Date.now()/1000)
        console.log('time now', (Date.now()/1000))
        console.log('registration deadline', registrationDeadline)
        if (isRegistrationOver){
            alert("Registration period over, can't add more clients.")
            return;
        }

        // Check is user has already been registered
        let alreadyRegistered = registered.includes(accounts[0])
        if (alreadyRegistered){
            alert("Cannot register twice, you have already registered to this job")
            return;
        }

        // Minimum payment amount
        const amountToPay = await jobsdatabase.methods.holdingFee().call()
        // register dataset to job
        //arg: _jobID, _datasetHash
        await jobsdatabase.methods.registerDatasetOwner(this.props.job, this.state.ipfsHash).send({from: accounts[0], value: parseInt(amountToPay)})
        .on('transactionHash', (hash) =>{
            console.log(hash);
            this.setState({transactionHash:hash})
        })
        .on('error', async (error, receipt) => {
            console.log(error);
            if (receipt) {
                console.log(receipt["transactionHash"])
            }
        })


    }


    render() {
        //const { formErrors } = this.state;
        return (

        <form onSubmit={this.handleSubmit}>
            <div className="container">
                <div className='subContainer'>
                    <h2>Register dataset to job</h2>
                    <hr />
                    <label>
                    <b>Dataset IPFS hash</b>:
                    <input name="ipfsHash" type="text" value={this.state.ipfsHash} onChange={this.handleChange} />
                    </label>

                    <input type="submit" value="Register" className="register"/>
                </div>
            </div>
        </form>
        )
    }
}