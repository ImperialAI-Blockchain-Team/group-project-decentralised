import React from "react";
import "./model.css"
import {UploadModelForm} from "../../forms/model.js"

export class RegisterModelPage extends React.Component {

    render() {
        return (
            <div className='form-container'>
                <UploadModelForm />
            </div>
        )
    }
}