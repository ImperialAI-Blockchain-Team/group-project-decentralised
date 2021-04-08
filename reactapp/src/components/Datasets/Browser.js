import React from "react";
import "./Browser.css"
import web3 from "../../web3";
import datasetdatabase from "../../datasetdatabase";

export class DataBrowser extends React.Component {

    constructor(props) {

        super(props);
        this.state = {
            searchValue: '',
            ethAddress: '',
            numberOfDatasets: -1,
            datasetHashList: [],
            datasetList: [],
            renderedDatasetList: []
            }
        this.handleOnKeyUp = this.handleOnKeyUp.bind(this);

        // call smart contract to render Datasets
        this.getNumberOfDatasets()
        .then(this.getDatasetList, (err) => {alert(err)})
        .then(this.renderDatasets, (err) => {alert(err)});
    }

    getNumberOfDatasets = async () => {
        let numberOfDatasets = await datasetdatabase.methods.getNumberOfs().call();
        this.setState({numberOfDatasets: numberOfDatasets});
        return new Promise((resolve, reject) => {
            if (numberOfDatasets != -1) {
                resolve(numberOfDatasets);
            } else {
                reject(Error("Can't connect to datasetdatabase smart contract."))
            }
        })
    }

    getDatasetList = async (numberOfDatasets) => {
        var newDatasetHashList = [];
        var newDatasetList = [];
        for (var i=0; i<numberOfDatasets; i++) {
            const ipfsHash = await datasetdatabase.methods.hashes(i).call();
            const dataset = await datasetdatabase.methods.datasets(ipfsHash).call();
            newDatasetHashList.push(ipfsHash+ '---');
            newDatasetList.push(dataset);
        }
        newDatasetHashList.reverse();
        newDatasetList.reverse();
        this.setState({datasetHashList: newDatasetHashList})
        this.setState({datasetList: newDatasetList})

        return new Promise((resolve, reject) => {
            resolve(newDatasetList);
        })
    }


    renderDatasets = async (datasetList) => {
        const subDatasetList = datasetList.filter(dataset => {
            return dataset['description'].toLowerCase().startsWith(this.state.searchValue)
        })

        const renderedDatasets = await subDatasetList.map(dataset => {
            return (
            <div className="DatasetContainer">
                <div className="subDatasetContainer">
                    <p><b>Owner</b>: {dataset['owner']}</p>
                    <p><b>Name</b>: not implemented{}</p>
                    <p><b>Description</b>: {dataset['description']}</p>
                    <p><b>Creation Date</b>: {new Date(dataset['time']*1000).toLocaleDateString()}</p>
                </div>
            </div>
            )
        })
        this.setState({renderedDatasetList: renderedDatasets});
    }

    handleOnKeyUp = async (event) => {
        const target = event.target;
        const name = target.name;
        await this.setState({searchValue: event.target.value});
        this.renderDatasets(this.state.datasetList);
    }

    render() {

        return (
            <div className="container">
                <div className="searchBarContainer">
                    <input type="text" id="myInput" onKeyUp={this.handleOnKeyUp} placeholder="Search dataset (by description)" />
                </div>
                <div className="headerContainer">
                    <p id="numberOfDatasets">{this.state.numberOfDatasets} Datasets already uploaded to the system</p>
                    <hr />
                </div>
                <div className="datasetListContainer">
                    <tr>
                        <p>{this.state.renderedDatasetList}</p>
                    </tr>
                </div>
            </div>
        )
    }
}