import React from "react";
import "./dragZone.css"

export class DragZone extends React.Component {

    handleDragOver(event) {
        event.preventDefault();
    }

    handleDragEnter(event) {
        event.preventDefault();
    }

    handleDragLeave(event) {
        event.preventDefault();
    }

    handleFileDrop(event) {
        event.preventDefault();
        const files = event.dataTransfer.files;
        console.log(files);
    }

    render() {
        return (
            <div className="dropZoneContainer"
                onDragOver={this.handleDragOver}
                onDragEnter={this.handleDragEnter}
                onDragLeave={this.handleDragLeave}
                onDrop={this.handleDrop}>
                <div className="dropMessage">
                    <div className="uploadIcon"></div>
                    Drag & Drop files here or click to upload
                </div>
            </div>
        )
    }
}