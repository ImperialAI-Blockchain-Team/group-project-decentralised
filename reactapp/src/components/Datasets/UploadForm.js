import React from "react";
import { DragZone } from "./DragZone";
import "./UploadForm.css";
import ipfs from '../../ipfs'
import web3 from "../../web3";
import datasetdatabase from "../../datasetdatabase";

export class UploadDatasetForm extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            name: '',
            address: '',
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
            datasetdatabase.methods.register_dataset(this.state.ipfsHash, this.state.name, 'Objective').send({
                from: accounts[0]},
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
        return (
        <form onSubmit={this.handleSubmit}>
            <div className="container">
                <div className='subContainer'>
                    <h2>Register anonymised samples of your Dataset</h2>
                    <p>Please fill in this form to register your dataset.</p>
                    <hr />
                    <label>
                    <b>Dataset Name</b>:
                    <input name="name" type="text" value={this.state.name} onChange={this.handleChange} />
                    </label>
                    <label>
                    <b>Description</b>:
                    <input name="address" type="text" value={this.state.address} onChange={this.handleChange} />
                    </label>

                    <label>
                    <b>Dataset</b>:
                    <input name= "dataset" type = "file"
                               onChange = {this.captureFile}
                    />
                    </label>

                    <input type="submit" value="Register" className="register"/>

                </div>
            </div>
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
                        </tr>                  <tr>
                            <td>Tx # </td>
                            <td> : </td>
                            <td>{this.state.transactionHash}</td>
                        </tr>
                        </tbody>
                    </table>
        </form>

        )
    }
}
