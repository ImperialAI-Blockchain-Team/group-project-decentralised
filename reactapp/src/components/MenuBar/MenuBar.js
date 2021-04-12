import React from "react";
import {Link} from 'react-router-dom';
import "./MenuBar.css"

export class MenuBar extends React.Component {

    render() {

        return (
                <div class="navbar">
                    <Link to="/about">About</Link>
                    <Link to="/about">Sign Up</Link>
                    <Link to="/about">My Account</Link>
                    <div class="dropdown">
                        <button class="dropbtn">Register your Assets<i class="fa fa-caret-down"></i></button>
                        <div class="dropdown-content">
                            <Link to="/registermodel">Your Model</Link>
                            <a href="#">Your Dataset</a>
                        </div>
                    </div>
                    <div class="dropdown">
                        <button class="dropbtn">Explore <i class="fa fa-caret-down"></i></button>
                        <div class="dropdown-content">
                            <a href="#">Models</a>
                            <a href="#">Datasets</a>
                            <a href="#">Jobs</a>
                        </div>
                    </div>
                </div>
        )
    }
}