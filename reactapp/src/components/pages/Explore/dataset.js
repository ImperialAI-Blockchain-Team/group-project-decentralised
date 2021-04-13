import React from "react";
import "./dataset.css"
import {DatasetBrowser} from "../../browsers/dataset.js"

export class BrowseDatasetsPage extends React.Component {

    render() {
        return (
            <div className='form-container'>
                <DatasetBrowser />
            </div>
        )
    }
}