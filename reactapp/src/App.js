import React from "react";
import { Route, Switch, Redirect } from 'react-router-dom';
import "./App.css";
<<<<<<< HEAD
import {MenuBar} from "./components/menuBar/MenuBar.js";
=======
import {MenuBar} from "./components/MenuBar/MenuBar.js";
>>>>>>> db1c74363462cbac26fc14ffc930fcd03f2025ea
import {RegisterDatasetPage} from "./components/pages/RegisterAsset/dataset.js";
import {RegisterModelPage} from "./components/pages/RegisterAsset/model.js";
import {RegisterUserPage} from "./components/pages/SignUp/main.js";
import {BrowseDatasetsPage} from "./components/pages/Explore/dataset.js";
import {BrowseModelsPage} from "./components/pages/Explore/model.js";
import {JobForm} from "./components/job.js";

function App() {

  return (
    <body>
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
          <Route path='/create_job' component={JobForm}/>
          {/* <Route path='/browse_jobs' component={JobForm}/> */}
        </Switch>
      </div>
    </body>
  )
  }

 export default App;
