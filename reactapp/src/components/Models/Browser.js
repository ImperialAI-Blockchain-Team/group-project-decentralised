import React from "react";
import "./Browser.css"
import web3 from "../../web3";
import modeldatabase from "../../modeldatabase";

export class ModelBrowser extends React.Component {

    constructor(props) {

        super(props);
        this.state = {
            searchValue: '',
            ethAddress: '',
            numberOfModels: -1,
            modelHashList: [],
            modelList: [],
            renderedModelList: []
            }
        this.handleOnKeyUp = this.handleOnKeyUp.bind(this);

        // call smart contract to render models
        this.getNumberOfModels()
        .then(this.getModelList, (err) => {alert(err)})
        .then(this.renderModels, (err) => {alert(err)});
    }

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
            newModelHashList.push(ipfsHash+ '---');
            newModelList.push(model);
        }
        newModelHashList.reverse();
        newModelList.reverse();
        this.setState({modelHashList: newModelHashList})
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
                <div className="subModelContainer">
                    <p><b>Owner</b>: {model['owner']}</p>
                    <p><b>Name</b>: not implemented{}</p>
                    <p><b>Description</b>: {model['description']}</p>
                    <p><b>Creation Date</b>: {new Date(model['time']*1000).toLocaleDateString()}</p>
                </div>
            </div>
            )
        })
        this.setState({renderedModelList: renderedModels});
    }

    handleOnKeyUp = async (event) => {
        const target = event.target;
        const name = target.name;
        await this.setState({searchValue: event.target.value});
        this.renderModels(this.state.modelList);
    }

    render() {

        return (
            <div className="container">
                <div className="searchBarContainer">
                    <input type="text" id="myInput" onKeyUp={this.handleOnKeyUp} placeholder="Search model (by description)" />
                </div>
                <div className="headerContainer">
                    <p id="numberOfModels">{this.state.numberOfModels} models already uploaded to the system</p>
                    <hr />
                </div>
                <div className="modelListContainer">
                    <tr>
                        <p>{this.state.renderedModelList}</p>
                    </tr>
                </div>
            </div>
        )
    }
}