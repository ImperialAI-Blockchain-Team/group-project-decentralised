import React from "react";
import {Link} from 'react-router-dom';
import "./MenuBar.css"

export class MenuBar extends React.Component {

    render() {

        return (
                <div className="navbar">
                    <Link to="/about">About</Link>
                    <Link to="/sign_up">Sign Up</Link>
                    <div className="dropdown">
                        <button className="dropbtn">Register your Assets<i className="fa fa-caret-down"></i></button>
                        <div className="dropdown-content">
                            <Link to="/register_model">Your Model</Link>
                            <Link to="/register_dataset">Your Dataset</Link>
                        </div>
                    </div>
                    <div className="dropdown">
                        <button className="dropbtn">Explore <i className="fa fa-caret-down"></i></button>
                        <div className="dropdown-content">
                            <Link to="/browse_models">Models</Link>
                            <Link to="/browse_datasets">Datasets</Link>
                            <Link to="/browse_jobs">Jobs</Link>
                        </div>
                    </div>
                </div>
        )
    }
}