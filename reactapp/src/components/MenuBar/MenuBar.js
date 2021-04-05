import React from "react";
import "./MenuBar.css"
import {RegisterNodeForm} from "../FLNodes/RegisterForm.js";
import {UploadModelForm} from "../Models/UploadForm.js"

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
                    <button class="tablink" name="models" onClick={this.handleClick}>Models</button>
                    <button class="tablink" name="datasets" onClick={this.handleClick}>Datasets</button>
                    <button class="tablink" name="FLnodes" onClick={this.handleClick}>Federated Learning Nodes</button>
                </div>
                <div className="tabContent">
                    {(this.state.form === "models") ? <UploadModelForm />: null}
                    {(this.state.form === "datasets") ? <UploadModelForm /> : null}
                    {(this.state.form === "FLnodes") ? <RegisterNodeForm />: null}
                </div>
            </div>
        )
    }
}