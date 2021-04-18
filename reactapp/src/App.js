import React from "react";
import { Route, Switch, Redirect } from 'react-router-dom';
import "./App.css";
import {MenuBar} from "./components/menuBar/MenuBar.js";
import {RegisterDatasetPage} from "./components/pages/RegisterAsset/dataset.js";
import {RegisterModelPage} from "./components/pages/RegisterAsset/model.js";
import {RegisterUserPage} from "./components/pages/SignUp/main.js";
import {BrowseDatasetsPage} from "./components/pages/Explore/dataset.js";
import {BrowseModelsPage} from "./components/pages/Explore/model.js";
import {BrowseJobsPage} from "./components/pages/Explore/job";
import {JobForm} from "./components/forms/job.js";
import {ClientPackage} from "./components/package.js"

function App() {

  return (
    <div>
      <div className="header">
        <div className="title">
          <h1>Software Engineering Group Project</h1>
        </div>
      </div>

      <MenuBar />

      <div className="page-container">
        <Switch>
          <Route exact path='/'>
            <Redirect to="/about" />
          </Route>
          {/* <Route path='/about' component={}/> */}
          <Route path='/sign_up' component={RegisterUserPage}/>
          <Route path='/register_model' component={RegisterModelPage}/>
          <Route path='/register_dataset' component={RegisterDatasetPage}/>
          <Route path='/browse_models' component={BrowseModelsPage}/>
          <Route path='/browse_datasets' component={BrowseDatasetsPage}/>
          <Route path='/browse_jobs' component={BrowseJobsPage}/>
          <Route path='/test_components' component={ClientPackage}/>
          {/* <Route path='/create_job' component={JobForm}/> */}
          {/* <Route path='/browse_jobs' component={JobPage}/> */}
        </Switch>
      </div>
    </div>
  )
  }

 export default App;
