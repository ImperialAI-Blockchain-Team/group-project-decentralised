import React from "react";
import "./UploadForm.css";
import ipfs from '../../ipfs'
import web3 from "../../web3";
import modeldatabase from "../../modeldatabase";

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
            description: '',
            ipfsHash: null,
            buffer: '',
            ethAddress: '',
            transactionHash: '',
            txReceipt: '',
            displayTable: false,
            errors: []
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
        const errors = validate(this.state.name, this.state.description, this.state.buffer)
        if (errors.length > 0){
            this.setState({ errors });
            return;
        }
        //bring in user's metamask account address
        const accounts = await web3.eth.getAccounts();
        //obtain contract address from modelDatabase.js
        const ethAddress = await modeldatabase.options.address;
        this.setState({ethAddress});
        //save document to IPFS,return its hash#, and set hash# to state
        await ipfs.add(this.state.buffer, (err, ipfsHash) => {
            console.log(err, ipfsHash);
            //setState by setting ipfsHash to ipfsHash[0].hash
            this.setState({ipfsHash: ipfsHash[0].hash});

            // return the transaction hash from the ethereum contract
            modeldatabase.methods.register_model(this.state.ipfsHash, this.state.name, 'Objective').send({
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
        const { errors } = this.state;
        return (
        <form onSubmit={this.handleSubmit}>

            <div className="container">
                <div className='subContainer'>
                    <h2>Register your Model</h2>
                    <p>Please fill in this form to register your model.</p>
                    <hr />
                    <label>
                    <b>Model Name</b>:
                    <input name="name" type="text" value={this.state.name} onChange={this.handleChange} />
                    </label>
                    <label>
                    <b>Description</b>:
                    <input name="description" type="text" value={this.state.description} onChange={this.handleChange} />
                    </label>

                    <label>
                    <b>Model</b>:
                    <input name= "model" type = "file"
                               onChange = {this.captureFile}
                    />
                    </label>

                    <input type="submit" value="Register" className="register"/>

                </div>
                {errors.map(error => (
                    <p key={error}>Error: {error}</p>
                ))}
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
