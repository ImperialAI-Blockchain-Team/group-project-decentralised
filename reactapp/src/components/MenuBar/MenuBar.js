import React from "react";
import "./MenuBar.css"
import {RegisterNodeForm} from "../FLNodes/RegisterForm.js";

export class MenuBar extends React.Component {

    constructor() {
        super()
        this.state = {
            form: <RegisterNodeForm />
        }
    }

    handleClick = () => {
        // if (Math.random()>0.5) {
        //     this.setState({message: <p>yo1</p>})
        // } else {
        //     this.setState({message: <p>yo2</p>});
        // }
    };

    render() {
        return (
            <div className="mainBody">
                <div className="tab">
                    <button class="tablink" onClick={this.handleClick}>Models</button>
                    <button class="tablink" onClick={this.handleClick}>Datasets</button>
                    <button class="tablink" onClick={this.handleClick}>Federated Learning Nodes</button>
                </div>
                <div className="tabContent">
                    {this.state.form}
                </div>
            </div>
        )
    }
}