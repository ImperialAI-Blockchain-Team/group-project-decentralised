import React from "react";
import {Link} from 'react-router-dom';
import "./menuBar.css"

export class MenuBar extends React.Component {

    render() {

        return (
                <div class="navbar">
                    <Link to="/about">About</Link>
                    <Link to="/sign_up">Sign Up</Link>
                    <Link to="/my_account">My Account</Link>
                    <div class="dropdown">
                        <button class="dropbtn">Register your Assets<i class="fa fa-caret-down"></i></button>
                        <div class="dropdown-content">
                            <Link to="/register_model">Your Model</Link>
                            <Link to="/register_dataset">Your Dataset</Link>
                        </div>
                    </div>
                    <div class="dropdown">
                        <button class="dropbtn">Explore <i class="fa fa-caret-down"></i></button>
                        <div class="dropdown-content">
                            <Link to="/browse_models">Models</Link>
                            <Link to="/browse_datasets">Datasets</Link>
                            <Link to="/browse_jobs">Jobs</Link>
                        </div>
                    </div>
                </div>
        )
    }
}