import React from "react";
import "./model.css"
import modeldatabase from "../../contractInterfaces/modeldatabase";
import {Link} from 'react-router-dom';
import { Container } from "../helpers/Container";
import web3 from "../../contractInterfaces/web3";

export class ModelBrowser extends React.Component {

    constructor(props) {

        super(props);
        this.state = {
            searchValue: '',
            ethAddress: '',
            numberOfModels: 'Loading...',
            modelList: [],
            renderedModelList: (
                <div className="loadingCell">
                    <p><b> Loading ... </b></p>
                </div>
                ),
            modelInfo: this.browserIntroduction(),
            triggerText: "Create Job"
            }
        this.handleOnKeyUp = this.handleOnKeyUp.bind(this);

        // call smart contract to render models
        this.getNumberOfModels()
        .then(this.getModelList, (err) => {alert(err)})
        .then(this.renderModels, (err) => {alert(err)});
    }

    browserIntroduction = () => {
        return (
            <div className="modelInfo">
                <h3>Click on a model to display additional info! </h3>
            </div>
        )
    };

    getNumberOfModels = async () => {
        let numberOfModels = await modeldatabase.methods.getNumberOfModels().call();
        this.setState({numberOfModels: numberOfModels});
        return new Promise((resolve, reject) => {
            if (numberOfModels != -1) {
                resolve(numberOfModels);
            } else {
                reject(Error("Can't connect to ModelDatabase smart contract."))
            }
        })
    }

    getModelList = async (numberOfModels) => {
        var newModelHashList = [];
        var newModelList = [];
        for (var i=0; i<numberOfModels; i++) {
            const ipfsHash = await modeldatabase.methods.hashes(i).call();
            const model = await modeldatabase.methods.models(ipfsHash).call();
            model['ipfsHash'] = ipfsHash;
            model['index'] = i;
            newModelList.push(model);
        }
        newModelList.reverse();
        this.setState({modelList: newModelList})

        return new Promise((resolve, reject) => {
            resolve(newModelList);
        })
    }


    renderModels = async (modelList) => {
        const subModelList = modelList.filter(model => {
            return model['name'].toLowerCase().startsWith(this.state.searchValue)
        })
        const { triggerText } = this.state.triggerText;
        const onSubmit = (event) => {
            event.preventDefault(event);
            console.log(event.target.name.value);
            console.log(event.target.email.value);
        };
        const renderedModels = await subModelList.map(model => {
            return (
            <div className="modelContainer">
                <p><b>Owner</b>: {model['owner']}</p>
                <p><b>Name</b>: {model['name']}</p>
                <p><b>Objective</b>: {model['objective']}</p>
                <p><b>Creation Date</b>: {new Date(model['time']*1000).toLocaleDateString()}</p>
                <p><button className="moreInfoButton" name={model['index']} onClick={this.handleClick}>More Information</button>
                <Container triggerText={triggerText} model={model['ipfsHash']} />
                {/* <button id='like'>like</button> */}
                </p>
            </div>
            )
        })
        this.setState({renderedModelList: renderedModels});
    }

    handleClick = async (event) => {
        let model_index = Number(event.target.name);
        let modelInfo = (
            <div className="modelInfo">
                <p><b>Owner</b>:<br/> {this.statemodelList[model_index]['owner']}</p>
                <p><b>Name</b>:<br/> {this.statemodelList[model_index]['name']}</p>
                <p><b>Objective</b>:<br/> {this.statemodelList[model_index]['objective']}</p>
                <p><b>Description</b>:<br/> {this.statemodelList[model_index]['description']}</p>
                <p><b>Data Requirements</b>:<br/> {this.statemodelList[model_index]['dataRequirements']}</p>
            </div>
            )
        this.setState({modelInfo: modelInfo})
    }

    handleOnKeyUp = async (event) => {
        const target = event.target;
        const name = target.name;
        await this.setState({searchValue: event.target.value});
        this.renderModels(this.state.modelList);
    }

    render() {
        return (
            <div className="pageContainer">
                <div className="headerContainer">
                    <div className="searchBarContainer">

                        <input type="text" id="myInput" onKeyUp={this.handleOnKeyUp} placeholder="Search model by name" />
                    </div>
                    <p id="numberOfModels">{this.state.numberOfModels} models already uploaded to the system</p>
                    <hr />
                </div>
                <div className="resultContainer">
                    <tr>
                        {this.state.renderedModelList}
                    </tr>
                </div>
                <div className="modelInfoContainer">
                    {this.state.modelInfo}
                </div>
            </div>
        )
    }
}