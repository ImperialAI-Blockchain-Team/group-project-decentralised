import React from "react";
import "./dataset.css"
import {ModelBrowser} from "../../browsers/model.js"

export class BrowseModelsPage extends React.Component {

    render() {
        return (
            <div className='form-container'>
                <ModelBrowser />
            </div>
        )
    }
}