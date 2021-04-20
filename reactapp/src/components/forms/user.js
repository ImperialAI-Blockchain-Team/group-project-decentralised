import React from "react";
import "./user.css";

import web3 from "../../contractInterfaces/web3";
import registrydatabase from "../../contractInterfaces/registrydatabase"
import modelDatabase from "../../abi/abi";


function validate(username, email, type){

    const errors = [];

    if (username.length === 0) {
        errors.push("Name can't be empty");
    }
    if (email.length === 0) {
        errors.push("Email can't be empty");
    }
    if (type.length === 0){
        errors.push("Please choose a type")
    }

    let re = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;

    if ( re.test(email) ) {
        // this is a valid email address
    }
    else {
        errors.push("Enter a valid email")
    }

    return errors
}

export class RegisterUserForm extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            name: '',
            email: '',
            type: [],
            data_scientist: false,
            data_owner:false,
            account : '',
            ethAddress:'',
            index : ''
        };
        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }

    handleChange(event) {
        const target = event.target;
        const name = target.name;
        this.setState({[name]: event.target.value});
    }

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


    handleSubmit = async (event) => {
        event.preventDefault();

        const errors = validate(this.state.name, this.state.email, this.state.type)
        if (errors.length > 0){
            alert(JSON.stringify(errors));
            return;
        }
        //alert("Your details have been submitted")

        localStorage.setItem(JSON.stringify(this.state.name),JSON.stringify(this.state.address))


        //bring in user's metamask account address
        const accounts = await web3.eth.getAccounts();
        this.setState({account: accounts[0]})

        // First check if user already registered
        let isUser = await registrydatabase.methods.isUser(accounts[0]).call();
        if (isUser){
            alert("Already registered, cannot register again")
            return;
        }

        //const userNames = await registrydatabase.methods.arrNames().call()
        //let nameExists = userNames.includes(this.state.name);
        //if (nameExists){
        //    alert("Data name already taken, choose another name");
        //    return;
        //}

        //obtain contract address from registrydatabase.js
        const ethAddress = await registrydatabase.options.address;

        this.setState({ethAddress});
        console.log(accounts)
        // return the transaction hash from the ethereum contract
        await registrydatabase.methods.insertUser(this.state.name, this.state.data_scientist, this.state.data_owner).send({from: accounts[0]})
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

        //this.setState({index})
        console.log(JSON.stringify(this.state))

    }


    handleCheckChange = (event) => {
        // to find out if it's checked or not; returns true or false
        const checked = event.target.checked;


        // to get the checked value
        const checkedValue = event.target.value;

        if (checkedValue == "Data Scientist") {
            if (checked) {
                this.state.data_scientist = true;
            }else{
                this.state.data_scientist = false;
            }
        }

        if (checkedValue == "Data Owner") {
            if (checked) {
                this.state.data_owner = true;
            }else{
                this.state.data_owner = false;
            }
        }

        this.state.type.push(JSON.stringify(checkedValue));

    }

    render() {
        return (
        <form onSubmit={this.handleSubmit2}>
            <div className="container">
                <div className='sub-container'>
                    <h2>Register here</h2>
                    <p>Please fill in this form.</p>
                    <hr />

                    <label>
                    <b>Username</b>:
                    <input name="name" id ="name-input" type="text" value={this.state.name} onChange={this.handleChange} />
                    </label>
                    <label>
                    <b>Email Address</b>:
                    <input name="email" id ="email-input" type="text" value={this.state.email} onChange={this.handleChange} />
                    </label>
                    <label>
                    <b> User Type</b>:
                    </label>
                    <br></br>
                    <br></br>
                    <input type="checkbox" id="checkbox1" data-testid="check1"value = "Data Scientist" onChange={this.handleCheckChange}/><span>Data Scientist</span>
                    &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
                    <input type="checkbox" id="checkbox2" data-testid="check2"value = "Data Owner" onChange={this.handleCheckChange}/><span>Data Owner</span>

                    <br></br>
                    <br></br>
                    <button type = "button" onClick={this.handleSubmit}>Register</button>
                </div>
            </div>

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
