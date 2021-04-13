import React from "react";
import "./main.css"
import {RegisterUserForm} from "../../forms/user.js"

export class RegisterUserPage extends React.Component {

    render() {
        return (
            <div className='form-container'>
                <RegisterUserForm />
            </div>
        )
    }
}