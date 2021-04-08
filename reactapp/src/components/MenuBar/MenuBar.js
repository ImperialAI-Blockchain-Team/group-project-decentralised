import React from "react";
import "./MenuBar.css"
import {RegisterNodeForm} from "../FLNodes/RegisterForm.js";
import {UploadModelForm} from "../Models/UploadForm.js";
import {UploadDatasetForm} from "../Datasets/UploadForm.js";
import {ModelBrowser} from "../Models/Browser.js";
import {DataBrowser} from "../Datasets/Browser.js"

export class MenuBar extends React.Component {

    constructor() {
        super()
        this.state = {
            form: "models"
        }
        this.handleClick = this.handleClick.bind(this);
    }

    handleClick(event) {
        const target = event.target;
        this.setState({form: target.name})
        }

    render() {

        return (
            <div className="mainBody">
                <div className="tab">
                    <button class="tablink" name="register" onClick={this.handleClick}>Register</button>
                    <button class="tablink" name="models" onClick={this.handleClick}>ML Models</button>
                    <button class="tablink" name="datasets" onClick={this.handleClick}>Datasets</button>
                    <button class="tablink" name="FLnodes" onClick={this.handleClick}>FL Aggregators</button>
                    <button class="tablink" name="search_model" onClick={this.handleClick}>Search Model</button>
                    <button class="tablink" name="search_dataset" onClick={this.handleClick}>Search Dataset</button>
                </div>
                <div className="tabContent">
                    {(this.state.form === "models") ? <UploadModelForm />: null}
                    {(this.state.form === "datasets") ? <UploadDatasetForm /> : null}
                    {(this.state.form === "FLnodes") ? <RegisterNodeForm />: null}
                    {(this.state.form === "search_model") ? <ModelBrowser />: null}
                    {(this.state.form === "search_dataset") ? <DataBrowser />: null}
                </div>
            </div>
        )
    }
}