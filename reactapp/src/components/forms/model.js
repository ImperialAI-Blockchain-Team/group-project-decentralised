import React from "react";
import "./model.css";
import ipfs from '../../ipfs'
import modelDatabase from "../../contractInterfaces/modeldatabase";
import registrydatabase from "../../contractInterfaces/registrydatabase";
import web3 from "../../contractInterfaces/web3";
import {Link} from 'react-router-dom';
import datasetdatabase from "../../contractInterfaces/datasetdatabase";

const getRevertReason = require('eth-revert-reason')

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

export class UploadModelForm extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            name: '',
            objective: '',
            description: 'What do you want to achieve? What is the purpose of your model? What would you need to succeed?',
            dataRequirements: '',
            ipfsHash: null,
            buffer: '',
            ethAddress: '',
            transactionHash: '',
            txReceipt: '',
            displayTable: false,
            formErrors: [],
            contractError: '',
        };
        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
        this.captureFile = this.captureFile.bind(this);
        this.convertToBuffer = this.convertToBuffer.bind(this);
    }

    handleChange(event) {
        const target = event.target;
        const name = target.name;
        this.setState({[name]: event.target.value});
    }

    handleSubmit = async (event) => {
        event.preventDefault();
        // Check if there are any form errors,
        const formErrors = validate(this.state.name, this.state.description, this.state.buffer)
        // If there is return and display errors
        if (formErrors.length > 0){
            this.setState({ formErrors });
            return;
        }
        //bring in user's metamask account address
        const accounts = await web3.eth.getAccounts();

         // First check if user is registered
        let isScientist = await registrydatabase.methods.isDataScientist(accounts[0]).call({from : accounts[0]});
        console.log(isScientist)
        if (!isScientist){
            alert("Not registered as data scientist need to register as data scientist first")
            return;
        }

        const names = await modelDatabase.methods.arrNames().call()

        let nameExists = names.includes(this.state.name);
        if (nameExists){
            alert("Data name already taken, choose another name");
            return;
        }

        //obtain contract address from modelDatabase.js
        const ethAddress = await modelDatabase.options.address;
        this.setState({ethAddress});

        //save document to IPFS,return its hash#, and set hash# to state
        await ipfs.add(this.state.buffer, (err, ipfsHash) => {
            console.log(err, ipfsHash);
            //setState by setting ipfsHash to ipfsHash[0].hash
            this.setState({ipfsHash: ipfsHash[0].hash});

            // return the transaction hash from the ethereum contract
            modelDatabase.methods.registerModel(this.state.ipfsHash,
                                                this.state.name,
                                                this.state.objective,
                                                this.state.description,
                                                this.state.dataRequirements).send({from: accounts[0]})
                .on('transactionHash', (hash) =>{
                    console.log(hash);
                    this.setState({transactionHash:hash})
                })
                .on('error', async (error, receipt) => {
                    console.log(error);
                    this.setState({contractError: 'Contract Error: Model already registered'})
                    if (receipt) {
                        console.log(receipt["transactionHash"])
                        //let txHash = receipt["transactionHash"]
                        //let blockNum = receipt["blockNumber"]
                        //console.log(await getRevertReason(txHash,'ropsten'))
                    }
                })
        })

    };

    //Take file input from user
    captureFile = (event) => {
        event.stopPropagation();
        event.preventDefault();
        const file = event.target.files[0];
        let reader = new window.FileReader();
        reader.readAsArrayBuffer(file);
        reader.onloadend = () => this.convertToBuffer(reader);
    };

    //Convert the file to buffer to store on IPFS
    convertToBuffer = async (reader) => {
        //file is converted to a buffer for upload to IPFS
        const buffer = await Buffer.from(reader.result);
        //set this buffer-using es6 syntax
        this.setState({buffer});
    };

    onClick = async () => {
        try{
            this.setState({blockNumber:"waiting.."});
            this.setState({gasUsed:"waiting..."});
            await web3.eth.getTransactionReceipt(this.state.transactionHash,
                (err, txReceipt)=>{console.log(err,txReceipt);
                this.setState({txReceipt});
            });
        } catch(error){console.log(error)}
        this.setState({displayTable:true})
    }


    render() {
        const { formErrors } = this.state;
        const { contractError } = this.state;
        return (

        <form onSubmit={this.handleSubmit}>
            <div className="container">
                <div className='sub-container'>
                    <h2>Register your Model</h2>
                    <p>Please fill in this form to register your model. <br /> <br />
                    After clicking <b>submit</b>, your model will be displayed in the <b>Explore</b> tab.
                    Data Owners can then register interest in your model! If your model is popular enougth,
                    you can then create a training configuration!
                    </p>
                    <hr />
                    <label>
                    <b>Model Name</b>:
                    <input name="name" type="text" value={this.state.name} onChange={this.handleChange}
                    placeholder="Give your model a name."
                    />
                    </label>
                    <label>
                    <b>Objective</b>:
                    <input name="objective" type="text" value={this.state.objective} onChange={this.handleChange}
                    placeholder="Describe in a few words the overall aim of your model."
                    />
                    </label>
                    <label>
                    <b>Description</b>:
                    <textArea name="description" type="text" onChange={this.handleChange}
                    placeholder="What do you want to achieve with your model? How much data would you need to fully train it?"
                    />
                    </label>
                    <label>
                    <b>Data Requirements</b>:
                    <textArea name="dataRequirements" type="text" onChange={this.handleChange}
                    placeholder="In what format does your model require the data to be in?
                    Please describe each attribute (data type, are missing values acceptable etc.) and the potential preprocessing required."
                    />
                    </label>
                    <label>
                    <b>Model: </b><br/>
                    <p>Your model must inherit from <b>torch.nn.Module</b>.
                    Additional classes relative to dataloading, training and testing must also be implemented. <br />
                    Please see the template for the full details.
                    </p>
                    <Link to="/model.py" target="_blank" download>
                        <p>download template</p>
                    </Link>
                    <input name= "model" type = "file" accept=".py"
                               onChange = {this.captureFile}
                    />
                    </label>

                    <input type="submit" value="Register" className="register"/>

                </div>
                {formErrors.map(error => (
                    <p key={error}>Error: {error}</p>
                ))}
                <p> {contractError} </p>

            </div>

            {!this.state.displayTable && <div className="center">
                <button onClick={this.onClick}>
                    {'Get Receipt'}
                </button>
            </div>}

            {this.state.displayTable && <div className="center">
                <table bordered responsive>
                    <thead>
                        <tr>
                            <th>Tx Receipt Category</th>
                            <th> </th>
                            <th>Values</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>IPFS Hash stored on Ethereum</td>
                            <td> : </td>
                            <td>{this.state.ipfsHash}</td>
                        </tr>
                        <tr>
                            <td>Ethereum Contract Address</td>
                            <td> : </td>
                            <td>{this.state.ethAddress}</td>
                        </tr>
                        <tr>
                            <td>Tx # </td>
                            <td> : </td>
                            <td>{this.state.transactionHash}</td>
                        </tr>
                    </tbody>
                </table>
            </div>}
        </form>
        )
    }
}
