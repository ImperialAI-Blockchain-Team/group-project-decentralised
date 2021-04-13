import React from "react";
import "./model.css"
import modeldatabase from "../../contractInterfaces/modeldatabase";
import {Link} from 'react-router-dom';

export class ModelBrowser extends React.Component {

    constructor(props) {

        super(props);
        this.state = {
            searchValue: '',
            ethAddress: '',
            numberOfModels: -1,
            modelList: [],
            renderedModelList: (
                <div className="loadingCell">
                    <p><b> Loading ... </b></p>
                </div>
                ),
            modelInfo: this.browserIntroduction()
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
            return model['description'].toLowerCase().startsWith(this.state.searchValue)
        })

        const renderedModels = await subModelList.map(model => {
            return (
            <div className="modelContainer">
                <p><b>Owner</b>: {model['owner']}</p>
                <p><b>Name</b>: not implemented{}</p>
                <p><b>Description</b>: {model['description']}</p>
                <p><b>Creation Date</b>: {new Date(model['time']*1000).toLocaleDateString()}</p>
                <p><button className="moreInfoButton" name={model['ipfsHash']} onClick={this.handleClick}>More Information</button>
                <Link to='create_job' id='jobButton'>Create Job</Link>
                {/* <button id='like'>like</button> */}
                </p>
            </div>
            )
        })
        this.setState({renderedModelList: renderedModels});
    }

    handleClick = async (event) => {

        let modelInfo = (
            <div className="modelInfo">
                <p><b>Info1</b>: something</p>
                <p><b>Info2</b>: something</p>
                <p><b>Info3</b>: something</p>
                <p><b>Info3</b>: something</p>
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

                        <input type="text" id="myInput" onKeyUp={this.handleOnKeyUp} placeholder="Search model (by description)" />
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