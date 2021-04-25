# Requirements

1. Create a wallet on MetaMask (https://metamask.io/download.html) and install the Chrome extension of MetaMask. Create two accounts, one for a data scientist, one for a hospital
2. In MetaMask choose the Ropsten Test Network on which our dApp is running. ETH on this network does not have a value. The network servers for development testing and production use cases
3. Get 1 ETH on https://faucet.ropsten.be/ for each account

4. Clone the code repository from https://gitlab.doc.ic.ac.uk/pd720/group-project.git
5. Install Poetry (https://python-poetry.org/docs/) \
   osx / linux / bashonwindows install instructions 
    ```properties
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
    ```

    windows powershell install instructions 
    ```properties
    (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
    ```

6. In the root directory of the repo run
   ```properties
   poetry update
   ``` 
7. Install npm

# Launching the dApp
Please read through the whole readme before launching the dApp.

To simulate the federated learning process locally do the following:

1. Start the frontend of the dApp in **localhost:3000** with
    ```properties
    cd reactapp
    run npm install
    run npm start
    ```
    If it is not launched in Chrome, open **localhost:3000** in Chrome.
    Click on the MetaMask extension in Chrome and connect your account to the site, choose the Ropsten network, do not close this terminal

1. Perform **steps 1-6** on the data scientist track using your data scientist's Ethereum account and **steps 1-5** on the hospital's track using your hospital's Ethereum account. You can easily swith accounts in MetaMask by clicking on the right upper corner icon

2. Run the flask backend which is used for the federated learning process. It has to run before a data scientist starts the training of a federated learning job on the frontend, i.e., when clicking **Start Training** on a job after performing tasks 1-6 on the data scientist track. Use a second terminal for this

    ```properties
    cd reactapp/FlowerBackend
    poetry shell
    flask run
    ```
3. Click on **Start Training** through the data scientist's account before the **Training Start Deadline**
4. A flower server will be started and you can perform **steps 6-8** of the hospital track

MISSING: More information about training and end of training




## Data scientist track
1.  Registration: Navigate to **Sign Up** and insert your details, select **Data scientist**, click on **Register**, MetaMask will pop up and ask you for confirming the contract interaction
2.  Registration of a model: Navigate to **Register your Assets** and click on **Your Model**, fill out the form and upload your model. It must inherit from torch.nn.Module. Additional classes relative to dataloading, training and testing must also be implemented. Please see the **template** for the full details
3.  Track the number of hospitals interested in contributing to your model, you can find it in **Explore** > **Models** under **More Information** of your model
4.  Register a job for your model in **Explore** > **Models** at **Create Job**. Define the duration of the registration phase during which hospitals can register for the job and a minimal number of clients you would need for the job. Complete the form and upload the test dataset corresponding to your model on which the evaluation of the federated learning training will be done
5. If you click **Register** MetaMask will pop up again and ask you to confirm a payment. You need to pay the bounty that you specified in the job form and a job creation fee of 1000000 Wei. The bounty serves as an incentive for hospitals to participate in training. The bounty is split over the participating hospitals based on their training accuracies. The creation fee you get back, see step 
6.  When a hospital with a dataset registeres for your job, you can decide whether it has suitable data for your model and add the data set to the allow list of the job by clicking **Add** in **Explore** > **Jobs** in **More Information** of your job
7.  When the registration deadline ends and you added more than the minimal number of clients specified in the job to the allow list, you can initiate the training process by clicking **Start Training**

## Hospital track
1. User Registration: Navigate to **Sign Up** and insert your details, select Hospital, click on **Register**, MetaMask will pop up and ask you for confirming the contract interaction
2. Registration of a dataset: Navigate to **Register your Assets** and click on **Your Dataset**, complete the form, click on **Register** and confirm the contract interaction in MetaMask
3. Browse the **Explore** > **Models** page and search for models your local dataset could contribute to. Register interest for suitable models by clicking **Register Interest** and confirm the smart contract interaction in MetaMask
4. Browse the **Explore** > **Jobs** page and search for federated learning jobs with **Status: Registration Phase**. If you have a suitable dataset available, that you have already registered, you can register your dataset for the federated learning job under **Register Dataset**. You have to submit the IPFS hash that was created when you registered your dataset. You can retrieve the hash under **Explore** > **Datasets**, click on **More Information** and copy the IPFS hash. By registering you commit to start training and pay a **holding fee** of 1000000 Wei. If you do not connect to the federated learning server you will not get the holding fee back
5. Note the **Training Start Deadline** of the job after which you are able to download the model by clicking on **Download Model**. Additionally you have to download the **Client Package** on top of the **Explore** > **Jobs** page
6. Follow the instructions in the client package and connect to the federated learning server
7. After successful training you will be compensated based on the accuracy of the model trained on your dataset
8. If the data scientist does not start training or not enough hospitals registered for the job you can withdraw the holding fee by clicking on **Withdraw Fee** after the **Training Start Deadline**