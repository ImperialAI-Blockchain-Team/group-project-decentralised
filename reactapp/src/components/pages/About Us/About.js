import React, { Component } from "react";

export class Main extends Component {
  render() {
    return (
        <div>
          <h1>Welcome to our Project Home! </h1>
          
          <div className="content">
              <p>The path to privacy preserving machine learning begins here.</p>
              <p>Perhaps you’re a Data Scientist who doesn’t have access to data in order to make advances in healthcare research. Or maybe you’re a medical institution looking for your data to be put to good use. Either way, you have come to the right place. </p>
              <p>Our team of software engineers have built an application that will enable you to embark on a journey of collaboration. Our goal is to facilitate collaboration in a privacy preserving manner, in order to train a model for predicting the need of ICU treatment for patients. </p>
              <p><strong>So how does it all work?</strong> Keep reading for a quick guide on how our software functions. </p>
              <ul>
                <li><p>To begin with, you will need a MetaMask account so as to allow you to interact with our Ethereum blockchain. </p></li>
                <li><p>To get started with the process, register yourself as a user by clicking on the Sign Up tab. Here, you’ll be asked to provide us with a few details, as well as indicate what type of user you are. </p></li>
                <li><p>Once you’re ready to start interacting, register your assets (either your model or sample dataset) by selecting the ‘Register your Assets’ tab. Upon successful registration of this asset to IPFS (InterPlanetary File System), you will be able to get a receipt for your transaction via Ethereum, along with the unique hash of your uploaded asset. </p></li>
                <li><p>As a Data Scientist, you can gather interest in your area of work, by ‘creating a job’ for your model. When creating this job, you will have to pay a certain bounty and job creation fee. You will also be required to submit a strategy and package for this task, at a later stage. As a Data Owner on the other hand, you can register your interest for available jobs by selecting ‘Jobs’ in the ‘Explore’ tab. A small amount of money will be charged to consolidate your interest.</p></li>
                <li><p>The job will begin as soon as the registration period ends, provided that there are a minimum number of viable users ready to take it on. Otherwise, you will be able to get your money back in an instant. </p></li>
                </ul>
                <br></br>
              <p>If all goes well, training can start! </p>
              <p>Your data will be dealt with <strong>safely </strong> and <strong>securely</strong> through our Flower Federated Learning Framework. In any particular training round, a Data Scientist can share the model for the Data owners to train on locally. The updated weights can then be sent back to initiate the following round of training. A final evaluation on test data will be used to calculate the improvement in the model’s accuracy. </p>
              <p>We will ensure the goals of the Data Scientist are met and that all Data Owners are rewarded proportionally – a win-win for both sides! </p>
              <p><i>Greater collaboration with a higher level of integrity?</i> Discover, contribute, and make a difference now by signing up to our Software using the tab above. </p>

          </div>
        </div>
    );
  }
}
 
