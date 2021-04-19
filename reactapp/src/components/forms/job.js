import React from "react";
import "./job.css";
import { Button } from 'react-bootstrap';
import axios from 'axios';
import jobsdatabase from "../../contractInterfaces/jobsdatabase";
import web3 from "../../contractInterfaces/web3";
import registrydatabase from "../../contractInterfaces/registrydatabase";
import ipfs from '../../ipfs';

export class JobForm extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            showDiv1: false,
            showDiv2: false,
            showX: false,
            showN: false,
            showU: false,
            showK: false,
            name: '',
            address: '',
            strategy: '',
            lr:'',
            epoch: '',
            batch_size: '',
            round:'',
            fraction_eval:'',
            fraction_fit:'',
            min_fit_clients:'',
            min_eval_clients:'',
            min_clients:'',
            failure:'',
            beta:'',
            slr:'',
            clr:'',
            da:'',
            distr:'',
            mean:'',
            std:'',
            ub:'',
            lb:'',
            gain:'',
            fan:'',
            linear: 'relu',
            slope:'',
            yes:false,
            no:false,
            registrationPeriod: '',
            bounty: '',
            transactionHash: '',
            minClients: '',
            testDatasetHash: '',
            strategyHash: null,
            buffer: ''
        };
        this.open1 = this.open1.bind(this);
        this.open2 = this.open2.bind(this);
        this.openn = this.openn.bind(this);
        this.openu = this.openu.bind(this);
        this.openx = this.openx.bind(this);
        this.openk = this.openk.bind(this);
        this.handleChange = this.handleChange.bind(this);
        this.handleReset = this.handleReset.bind(this);
        this.handleDefault1 = this.handleDefault1.bind(this);
        this.handleDefault2 = this.handleDefault2.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
        this.captureFile = this.captureFile.bind(this);
        this.convertToBuffer = this.convertToBuffer.bind(this);
    }

    handleChange(event) {
        const target = event.target;
        const name = target.name;
        this.setState({[name]: event.target.value});
    }

    handleReset() {
        this.setState({
            showDiv1: false,
            showDiv2: false,
            showX: false,
            showN: false,
            showU: false,
            showK: false,
            name: '',
            address: '',
            strategy: '',
            lr:'',
            epoch: '',
            batch_size: '',
            round:'',
            fraction_eval:'',
            fraction_fit:'',
            min_fit_clients:'',
            min_eval_clients:'',
            min_clients:'',
            failure:'',
            beta:'',
            slr:'',
            clr:'',
            da:'',
            distr:'',
            mean:'',
            std:'',
            ub:'',
            lb:'',
            gain:'',
            fan:'',
            linear: 'relu',
            slope:'',
            yes:false,
            no:false
    });}

    handleDefault1() {
        this.setState({
            epoch: 10,
            batch_size: 32,
            round:10,
            lr:0.001,
            fraction_eval:0.2,
            fraction_fit:0.3,
            min_fit_clients:2,
            min_eval_clients:2,
            min_clients:2,
            failure: true,
            yes:false,
            no:false

    });}
    handleDefault2() {
        this.setState({
            epoch: 10,
            batch_size: 32,
            round:10,
            lr:0.001,
            fraction_eval:0.2,
            fraction_fit:0.3,
            min_fit_clients:2,
            min_eval_clients:2,
            min_clients:2,
            failure: true,
            beta:0.99,
            slr:1e-1,
            clr:1e-1,
            da:1e-9,
            distr:'normal',
            showN: true,
            showX: false,
            showU: false,
            showK: false,
            mean:0,
            std:0.01,
            yes:false,
            no:false

    });}

    // handleSubmit = async (e) => {
    //     e.preventDefault();
    //     this.setState({
    //         yes: true,
    //         no: true
    //     });

    //     const data ={
    //         name: this.state.name,
    //         address: this.state.address,
    //         strategy: this.state.strategy,
    //         epoch: this.state.epoch,
    //         batch_size: this.state.batch_size,
    //         round:this.state.round,
    //         lr:this.state.lr,
    //         fraction_eval:this.state.fraction_eval,
    //         fraction_fit:this.state.fraction_fit,
    //         min_fit_clients:this.state.min_fit_clients,
    //         min_eval_clients:this.min_eval_clients,
    //         min_clients:this.state.min_clients,
    //         failure:this.state.failure,
    //         beta:this.state.beta,
    //         slr:this.state.slr,
    //         clr:this.state.clr,
    //         da:this.state.da,
    //         distr:this.state.distr,
    //         mean:this.state.mean,
    //         std:this.state.std,
    //         ub:this.state.ub,
    //         lb:this.state.lb,
    //         gain:this.state.gain,
    //         fan:this.state.fan,
    //         linear: this.state.linear,
    //         slope:this.state.slope,
    //     }
    //     axios.post("http://localhost:5000/start", data, {})
    //     .then(res => {
    //       console.log(data);
    //     }).catch(err => console.log("Error ", err));


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

    handleSubmit = async (event) => {
        event.preventDefault()
        this.setState({
            yes: true,
            no: true
        });
        console.log(this.props.model)
        //bring in user's metamask account address
        const accounts = await web3.eth.getAccounts();

        // First check if user is registered
        let isModelOwner = await jobsdatabase.methods.isSenderModelOwner(this.props.model).call({from : accounts[0]});
        console.log(isModelOwner)
        if (!isModelOwner){
            alert("Not model owner, only model owner can create job for model")
            return;
        }
        // Minimum payment amount
        const jobFee = await jobsdatabase.methods.jobCreationFee().call()
        const amountToPay = parseInt(this.state.bounty) + parseInt(jobFee)

        await ipfs.add(this.state.buffer)
            .then(res => {
            const hash = res[0].hash
            this.setState({testDatasetHash:hash});
            console.log('test dataset hash', this.state.testDatasetHash);
        })

        // Create strategy metadata
        const allowedKeys = [
                            "address",
                            "strategy",
                            "epoch",
                            "batch_size",
                            "round",
                            "fraction_eval",
                            "fraction_fit",
                            "min_fit_clients",
                            "min_eval_clients",
                            "min_clients",
                            "failure",
                            "beta",
                            "slr",
                            "clr",
                            "da",
                            "distr",
                            "mean",
                            "std",
                            "ub",
                            "lb",
                            "gain",
                            "fan",
                            "linear",
                            "slope",
                            "minClients"
                        ];
        const strategyMetadata = Object.keys(this.state)
            .filter(key => allowedKeys.includes(key))
            .reduce((obj, key) => {
                obj[key] = this.state[key];
                return obj;
            }, {});
        console.log(strategyMetadata);
        // Save strategy metadata to ipfs
        ipfs.add(Buffer.from(JSON.stringify(strategyMetadata)))
        .then(res => {
            const hash = res[0].hash
            this.setState({strategyHash:hash});
            console.log('strategy hash', this.state.strategyHash);
            return hash
        })
        .then(output => {
            // how to retrieve the strategy
            // const strategy = ipfs.files.cat(output);
            // console.log('retrieved strategy Hash', JSON.parse(strategy));
            // create job (string _modelIpfsHash, string _strategyHash, string _testDatasetHash, uint _minClients, uint _hoursUntilStart, uint _bounty)
            jobsdatabase.methods.createJob(this.props.model, output, this.state.testDatasetHash, parseInt(this.state.minClients),
            parseInt(this.state.registrationPeriod), parseInt(this.state.bounty)).send({from: accounts[0], value: amountToPay})
        .on('transactionHash', (hash) =>{
            console.log('transaction hash', hash);
            this.setState({transactionHash:hash})
        })
        .on('error', async (error, receipt) => {
            console.log(error);
            if (receipt) {
                console.log(receipt["transactionHash"])
                }
            })
        })
    }

    open1() {
        const { showDiv1 } = this.state;
        this.setState({
        showDiv1: !showDiv1,
            showDiv2: false,
            yes:false,
            no:false
    });}
    open2() {
        const { showDiv2 } = this.state;
        this.setState({
        showDiv2: !showDiv2,
            showDiv1: false,
            yes:false,
            no:false
    });}
    openn() {
        this.setState({
        showN: true,
            showU: false,
            showX: false,
            showK: false,
            yes:false,
            no:false
    });}
    openu() {
        this.setState({
        showU: true,
            showN: false,
            showX: false,
            showK: false,
            yes:false,
            no:false
    });}
    openx() {
        this.setState({
        showX: true,
            showU: false,
            showN: false,
            showK: false,
            yes:false,
            no:false
    });}
    openk() {
        this.setState({
        showK: true,
            showU: false,
            showX: false,
            showN: false,
            yes:false,
            no:false
    });}

    renderResetButton = () => {
        if (this.state.no) {
        return (
            <Button type="button" className="button" onClick={this.handleReset}>Reset</Button>
        );
        }
    }

    renderConfirmButton = () => {
        if (this.state.yes) {
        return (
            <form action="http://localhost:5000/flower" method="post"><button type="submit" className="button2">Confirm</button></form>
        )
        }
    }
    render() {

        return (

        <form onSubmit={this.handleSubmit}>
            <div className="container">
                <div className='subContainer'>
                    <h2>Create a job to train your model!</h2>
                    <p>Please fill in this form.</p>
                    <hr />

                    <label>
                    <b>Name</b>:
                    <input name="name" type="text" value={this.state.name} onChange={this.handleChange} />
                    </label>

                    <label>
                    <b>Registration Period (in hours)</b>:
                    <input name="registrationPeriod" type="text" value={this.state.registrationPeriod} onChange={this.handleChange} />
                    </label>

                    <label>
                    <b>Bounty (in Wei)</b>:
                    <input name="bounty" type="text" value={this.state.bounty} onChange={this.handleChange} />
                    </label>

                    <label>
                    <b>Minimum Number of Clients</b>:
                    <input name="minClients" type="text" value={this.state.minClients} onChange={this.handleChange} />
                    </label>

                    <label>
                    <b>Upload Test Dataset for Evaluation</b>:

                    <input name= "model" type = "file"
                               onChange = {this.captureFile}
                    />
                    </label>
                     <hr/>

                    <label>
                    <b>Aggregate Strategy (Choose one of below)</b>:
                    <br></br>
                    <br></br>
                    FedAvg:
                    <select name="strategy" value={this.state.strategy} onChange={this.handleChange}>
                        <option value="">--Mode Selection--</option>
                        <option value="datasize">datasize</option>
                        <option value="accuracy">accuracy</option>
                    </select>
                        <Button onClick={this.open1} disabled={this.state.strategy==''||this.state.strategy=='adagrad'||this.state.strategy=='yogi'||this.state.strategy=='adam'}>Hyperparameter Setting</Button>
                    <br></br>
                    <br></br>
                    FedOpt:
                    <select name="strategy" value={this.state.strategy} onChange={this.handleChange}>
                        <option value="">--Mode Selection--</option>
                        <option value="adagrad">adagrad</option>
                        <option value="yogi">yogi</option>
                        <option value="adam">adam</option>
                    </select>
                        <Button onClick={this.open2} disabled={this.state.strategy==''||this.state.strategy=='datasize'||this.state.strategy=='accuracy'}>Hyperparameter Setting</Button>
                    <br></br>
                    <br></br>
                    User-define:

                    </label>
                    { this.state.showDiv1 &&
                    <div className="dev1">
                        <button type="button" className="close" data-dismiss="alert" onClick={this.open1}>reselect strategy</button>
                        <button type="button" className="default" onClick={this.handleDefault1}>use default values</button>
                        <br />
                        epoch:
                        <input name="epoch" type="number" value={this.state.epoch} onChange={this.handleChange} /><br />
                        batch_size:
                        <input name="batch_size" type="number" value={this.state.batch_size} onChange={this.handleChange} /><br />
                        training round:
                        <input name="round" type="number" value={this.state.round} onChange={this.handleChange} /><br />
                        learning rate:
                        <input name="lr" type="number" value={this.state.lr} onChange={this.handleChange} /><br />
                        fraction_eval:
                        <input name="fraction_eval" type="number" value={this.state.fraction_eval} onChange={this.handleChange} /><br />
                        fraction_fit:
                        <input name="fraction_fit" type="number" value={this.state.fraction_fit} onChange={this.handleChange} /><br />
                        min_fit_clients:
                        <input name="min_fit_clients" type="number" value={this.state.min_fit_clients} onChange={this.handleChange} /><br />
                        min_eval_clients:
                        <input name="min_eval_clients" type="number" value={this.state.min_eval_clients} onChange={this.handleChange} /><br />
                        min_available_clients:
                        <input name="min_clients" type="number" value={this.state.min_clients} onChange={this.handleChange} /><br />
                        accept_failures:
                        <select name="failure" value={this.state.failure} onChange={this.handleChange}>
                            <option value="">--Selection--</option>
                            <option value="true">true</option>
                            <option value="false">false</option>
                        </select>
                        <br />
                    </div> }
                    { this.state.showDiv2 &&
                    <div className="dev2">
                        <button type="button" className="close" data-dismiss="alert" onClick={this.open2}>reselect strategy</button>
                        <button type="button" className="default" onClick={this.handleDefault2}>use default values</button>
                        <br />
                        epoch:
                        <input name="epoch" type="number" value={this.state.epoch} onChange={this.handleChange} /><br />
                        batch_size:
                        <input name="batch_size" type="number" value={this.state.batch_size} onChange={this.handleChange} /><br />
                        training round:
                        <input name="round" type="number" value={this.state.round} onChange={this.handleChange} /><br />
                        learning rate:
                        <input name="lr" type="number" value={this.state.lr} onChange={this.handleChange} /><br />
                        fraction_eval:
                        <input name="fraction_eval" type="number" value={this.state.fraction_eval} onChange={this.handleChange} /><br />
                        fraction_fit:
                        <input name="fraction_fit" type="number" value={this.state.fraction_fit} onChange={this.handleChange} /><br />
                        min_fit_clients:
                        <input name="min_fit_clients" type="number" value={this.state.min_fit_clients} onChange={this.handleChange} /><br />
                        min_eval_clients:
                        <input name="min_eval_clients" type="number" value={this.state.min_eval_clients} onChange={this.handleChange} /><br />
                        min_available_clients:
                        <input name="min_clients" type="number" value={this.state.min_clients} onChange={this.handleChange} /><br />
                        accept_failures:
                        <select name="failure" value={this.state.failure} onChange={this.handleChange}>
                            <option value="">--Selection--</option>
                            <option value="true">true</option>
                            <option value="false">false</option>
                        </select>
                        <br />
                        beta:
                        <input name="beta" type="number" value={this.state.beta} onChange={this.handleChange} /><br />
                        server-side learning rate:
                        <input name="slr" type="number" value={this.state.slr} onChange={this.handleChange} /><br />
                        client-side learning rate:
                        <input name="clr" type="number" value={this.state.clr} onChange={this.handleChange} /><br />
                        degree of adaptability:
                        <input name="da" type="number" value={this.state.da} onChange={this.handleChange} /><br />
                        Model initial weights:
                        <label><input type="radio" onClick={this.openn} name="distr" value="normal" checked={this.state.distr === 'normal'} onChange={this.handleChange} />Normal</label>
                        <label><input type="radio" onClick={this.openu} name="distr" value="uniform" checked={this.state.distr === 'uniform'} onChange={this.handleChange} />Uniform</label>
                        <label><input type="radio" onClick={this.openx} name="distr" value="xnormal" checked={this.state.distr === 'xnormal'} onChange={this.handleChange} />Xavier Normal</label>
                        <label><input type="radio" onClick={this.openx} name="distr" value="xuniform" checked={this.state.distr === 'xuniform'} onChange={this.handleChange} />Xavier Uniform</label>
                        <label><input type="radio" onClick={this.openk} name="distr" value="knormal" checked={this.state.distr === 'knormal'} onChange={this.handleChange} />Kaiming Normal</label>
                        <label><input type="radio" onClick={this.openk} name="distr" value="kuniform" checked={this.state.distr === 'kuniform'} onChange={this.handleChange} />Kaiming Uniform</label>
                        { this.state.showN &&
                        <div className="showN">
                            mean:
                            <input name="mean" type="number" value={this.state.mean} onChange={this.handleChange} /><br />
                            std:
                            <input name="std" type="number" value={this.state.std} onChange={this.handleChange} />
                        </div>}
                        { this.state.showU &&
                        <div className="showU">
                            upper bound:
                            <input name="ub" type="number" value={this.state.ub} onChange={this.handleChange} /><br />
                            lower bound:
                            <input name="lb" type="number" value={this.state.lb} onChange={this.handleChange} />
                        </div>}
                        { this.state.showX &&
                        <div className="showX">
                            gain:
                            <input name="gain" type="number" value={this.state.gain} onChange={this.handleChange} /><br />
                        </div> }
                        { this.state.showK &&
                        <div className="showK">
                            fan mode:
                            <select name="fan" value={this.state.fan} onChange={this.handleChange}>
                                <option value="">--Selection--</option>
                                <option value="fan_in">fan in</option>
                                <option value="fan_out">fan out</option>
                            </select><br />
                            nonlinearity:
                            <select name="linear" value={this.state.linear} onChange={this.handleChange}>
                                <option value="">--Selection--</option>
                                <option value="relu">relu</option>
                                <option value="leakyrelu">leaky relu</option>
                            </select>
                            <div className={this.state.linear}>slope:<input name="slope" type="number" value={this.state.slope} onChange={this.handleChange} /><br /></div>
                        </div>}
                    </div> }
                    <div>{this.renderConfirmButton()}</div>
                    <div>{this.renderResetButton()}</div>
                    <input type="submit" value="Register" className="register"/>
                </div>
            </div>
        </form>
        )
    }
}
