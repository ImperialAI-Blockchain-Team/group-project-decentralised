import React from "react";
import { DragZone } from "../dragZone/dragZone";
import "./dataset.css";
import ipfs from '../../ipfs'
import web3 from "../../contractInterfaces/web3";
import datasetdatabase from "../../contractInterfaces/datasetdatabase";
import registrydatabase from "../../contractInterfaces/registrydatabase";
import jobsdatabase from "../../contractInterfaces/jobsdatabase";

export class UploadDatasetForm extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            name: '',
            description: '',
            dataType: '',
            ipfsHash: null,
            buffer: '',
            ethAddress: '',
            transactionHash: '',
            txReceipt: ''
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
        //bring in user's metamask account address
        const accounts = await web3.eth.getAccounts();
        console.log(accounts)
        let isDataOwner = await registrydatabase.methods.isDataOwner(accounts[0]).call({from : accounts[0]});
        console.log(isDataOwner)
        if (!isDataOwner){
            alert("Not registered as data owner need to register as data-owner first")
            return;
        }

        //const names = await datasetdatabase.methods.arrNames().call()

        //let nameExists = names.includes(this.state.name);
        //if (nameExists){
        //    alert("Data name already taken, choose another name");
        //    return;
        //}

        //obtain contract address from storehash.js
        const ethAddress= await datasetdatabase.options.address;
        this.setState({ethAddress});
        //save document to IPFS,return its hash#, and set hash# to state
        await ipfs.add(this.state.buffer, (err, ipfsHash) => {
            console.log(err,ipfsHash);
            //setState by setting ipfsHash to ipfsHash[0].hash
            this.setState({ ipfsHash:ipfsHash[0].hash });
            // call Ethereum contract method "sendHash" and .send IPFS hash to etheruem contract
            // return the transaction hash from the ethereum contract
            datasetdatabase.methods.registerDataset(this.state.ipfsHash, this.state.name,
                                                this.state.description, this.state.dataType).send({from: accounts[0]},
                (error, transactionHash) => {
                console.log(transactionHash);
                this.setState({transactionHash});
            });
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

    render() {
        const { formErrors } = this.state;
        const { contractError } = this.state;
        return (

        <form onSubmit={this.handleSubmit}>
            <div className="container">
                <div className='sub-container'>
                <h2>Register your Dataset</h2>
                    <p>Please fill in this form to register your dataset. <br /> <br />
                    After clicking <b>submit</b>, your dataset will be displayed in the <b>Explore</b> tab.
                    You can then register interest in uploaded models and/or register for jobs with an active registration period.
                    Thereby you can contribute to training cutting-edge machine learning models that <b>revolutionise medicine</b>!
                    </p>
                    <hr />
                    <label>
                    <b>Dataset Name</b>:
                    <input name="name" id="name-input" type="text" value={this.state.name} onChange={this.handleChange}
                    placeholder="Give your dataset a name."
                    />
                    </label>
                    <label>
                    <b>Description</b>:
                    <textArea name="description" id ="description-input" type="text" value={this.state.description} onChange={this.handleChange}
                    placeholder="Please describe the data you own, how and when it was collected and the approximate dataset size."
                    />
                    </label>
                    <label>
                    <b>Data Type</b>:
                    <textArea input name="dataType" id = "dataType-input" type="text" value={this.state.dataType} onChange={this.handleChange}
                    placeholder="Please describe each attribute of your dataset precisely (data type, ranges, missing values etc.)."
                    />
                    </label>
                    <label>
                    <b>Synthetic Samples: </b><br/>
                    <p> Please upload a few samples of a synthetic dataset that resembles your local dataset.
                        The data should have the same format as your local dataset.
                    </p>
                    <input name= "dataset"  id = "fileUpload" data-testid = "file" type = "file"
                               onChange = {this.captureFile}
                    />
                    </label>

                    <input data-testid = "submit" type="submit" value="Register" className="register"/>

                </div>

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
