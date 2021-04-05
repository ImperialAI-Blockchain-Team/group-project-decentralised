import React from "react";
import "./RegisterForm.css";

export class RegisterNodeForm extends React.Component {

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
                    <h2>Register Federated Learning Node</h2>
                    <p>Please fill in this form.</p>
                    <hr />

                    <label>
                    <b>Node Name</b>:
                    <input name="name" type="text" value={this.state.name} onChange={this.handleChange} />
                    </label>
                    <label>
                    <b>Node IP Addres</b>:
                    <input name="address" type="text" value={this.state.address} onChange={this.handleChange} />
                    </label>
                    <input type="submit" value="Register" className="register"/>
                </div>
            </div>
        </form>
        )
    }
}
