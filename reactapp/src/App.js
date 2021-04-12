import React from "react";
import { Route, Switch, Redirect } from 'react-router-dom';
import "./App.css";
import {MenuBar} from "./components/MenuBar/MenuBar.js";
import {MainRegisterNodeForm} from "./components/Registration/MainRegisterForm.js";
import {UploadModelForm} from "./components/Models/UploadForm.js"

// const web3 = new Web3(Web3.givenProvider);
// const contractAddress = "0x659D25F48cd5d9Ee2b9f2cb243425D7df8cA2859";
// const storageContract = new web3.eth.Contract(simpleStorage, contractAddress);

function App() {

  return (
    <body>
      <div className="header">
        {/* <div className="uploadEthIcon"></div> */}
        <div className="title">
          <h1>Software Engineering Group Project</h1>
        </div>
      </div>
      <MenuBar />
      <Switch>
        <Route path='/register' component={MainRegisterNodeForm}/>
        <Route path='/registermodel' component={UploadModelForm}/>
      </Switch>
    </body>
  )
  }

 export default App;
