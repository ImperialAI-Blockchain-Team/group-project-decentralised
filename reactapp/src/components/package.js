import "./package.css";
import React from "react";
import {Link} from 'react-router-dom';

export class ClientPackage extends React.Component {

    render () {
        return (
            <div className="download-package-container">
                <p><b>Download this package and start making use of you data!</b></p>
                <Link to="/cw1.zip" target="_blank" download>
                    <img width='50' height='50' src= "../../public/folder_bis.png"/>
                </Link>
            </div>
        )
    }
}