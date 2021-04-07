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
            modelList: null
        }
        this.handleOnKeyUp = this.handleOnKeyUp.bind(this);

        // call smart contract to render model
        this.getModelList();
    }

    getModelList() {
        // const ethAddress = await modeldatabase.options.address;
        // this.setState({ethAddress});
        const numberOfModels = modeldatabase.methods.getNumberOfModels.call();
        this.setState({numberOfModels: numberOfModels})
    }

    handleOnKeyUp(event) {
        const target = event.target;
        const name = target.name;
        this.setState({searchValue: event.target.value});
    }

    render() {
        return (
            <div className="container">
                <div className="searchBarContainer">
                    <input type="text" id="myInput" onKeyUp={this.handleOnKeyUp} placeholder="Search model" />
                </div>
                <div className="modelListContainer">
                    <p>{this.state.numberOfModels}</p>
                </div>
            </div>
        )
    }
}