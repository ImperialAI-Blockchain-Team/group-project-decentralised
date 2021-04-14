import React from "react";
import "./dataset.css"
import {UploadDatasetForm} from "../../forms/dataset.js"

export class RegisterDatasetPage extends React.Component {

    render() {
        return (
            <div className='form-container'>
                <UploadDatasetForm />
            </div>
        )
    }
}