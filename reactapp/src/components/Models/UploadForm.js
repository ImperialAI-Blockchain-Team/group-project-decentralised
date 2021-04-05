import React from "react";
import { DragZone } from "./DragZone";
import "./UploadForm.css";

export class UploadModelForm extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            name: '',
            address: ''
        };
        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }

    handleChange(event) {
        const target = event.target;
        const name = target.name;
        this.setState({[name]: event.target.value});
    }

    handleSubmit(event) {

    }

    render() {
        return (
        <form onSubmit={this.handleSubmit}>
            <div className="container">
                <div className='subContainer'>
                    <h2>Register your Model</h2>
                    <p>Please fill in this form to register your model.</p>
                    <hr />
                    <label>
                    <b>Model Name</b>:
                    <input name="name" type="text" value={this.state.name} onChange={this.handleChange} />
                    </label>
                    <label>
                    <b>Description</b>:
                    <input name="address" type="text" value={this.state.address} onChange={this.handleChange} />
                    </label>

                    <div className="dragAndDropContainer">
                        <b>Drag and Drop your Model</b>:
                        <DragZone />
                    </div>

                    <input type="submit" value="Register" className="register"/>

                </div>
            </div>
        </form>
        )
    }
}
