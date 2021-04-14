import React from "react";
import "./dataset.css";
import DownloadLink from "react-download-link";
import ipfs from '../../ipfs';
import datasetdatabase from "../../contractInterfaces/datasetdatabase";

export class DatasetBrowser extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            searchValue: '',
            ethAddress: '',
            numberOfDatasets: '...',
            datasetList: [],
            renderedDatasetList: (
                                <div className="loadingCell">
                                    <p><b> Loading ... </b></p>
                                </div>
                                ),
            datasetInfo: this.browserIntroduction(),
            samples: null
            }
        this.handleOnKeyUp = this.handleOnKeyUp.bind(this);

        // call smart contract to render Datasets
        this.getNumberOfDatasets()
        .then(this.getDatasetList, (err) => {alert(err)})
        .then(this.renderDatasets, (err) => {alert(err)});
    }


    handleOnKeyUp = async (event) => {
        const target = event.target;
        const name = target.name;
        await this.setState({searchValue: event.target.value});
        this.renderDatasets(this.state.datasetList);
    }

    browserIntroduction = () => {
        return (
            <div className="datasetInfo">
                <h3>Click on a dataset to display additional info! </h3>
            </div>
        )
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
            dataset['ipfsHash'] = ipfsHash;
            newDatasetList.push(dataset);
        }
        newDatasetList.reverse();
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
                <div className="datasetContainer">
                    <p><b>Owner</b>: {dataset['owner']}</p>
                    <p><b>Name</b>: not implemented{}</p>
                    <p><b>Description</b>: {dataset['description']}</p>
                    <p><b>Creation Date</b>: {new Date(dataset['time']*1000).toLocaleDateString()}</p>
                    <p><button className="moreInfoButton" name={dataset['ipfsHash']} onClick={this.handleClick}>More Information</button></p>
            </div>
            )
        })
        this.setState({renderedDatasetList: renderedDatasets});

    }

    handleClick = async (event) => {
        const fileHash = event.target.name;
        await ipfs.files.get(fileHash, (err, files) => this.setState({'content': files[0]['content']}))

        let datasetInfo = (
            <div className="datasetInfo">
                <p><b>Info1</b>: something</p>
                <p><b>Info2</b>: something</p>
                <p><b>Info3</b>: something</p>
                <p><b>Info3</b>: something</p>
                <DownloadLink label="Download synthetic samples" filename="synthetic_samples"
                exportFile={() => this.state['content']}/>
            </div>
            )
        this.setState({datasetInfo: datasetInfo})
    }


    render() {
        return (
            <div className="pageContainer">
                <div className="headerContainer">
                    <div className="searchBarContainer">
                        <input type="text" id="myInput" onKeyUp={this.handleOnKeyUp} placeholder="Search dataset (by description)" />
                    </div>
                    <p id="numberOfDatasets">{this.state.numberOfDatasets} datasets already uploaded to the system</p>
                    <hr />
                </div>
                <div className="resultContainer">
                    <tr>
                        {this.state.renderedDatasetList}
                    </tr>
                </div>
                <div className="dataSampleContainer">
                    {this.state.datasetInfo}
                </div>
            </div>
        )
    }
}