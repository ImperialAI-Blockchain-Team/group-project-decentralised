import React from "react";
import "./dataset.css"
import {JobBrowser} from "../../browsers/job";

export class BrowseJobsPage extends React.Component {

    render() {
        return (
            <div className='form-container'>
                <JobBrowser />
            </div>
        )
    }
}