import subprocess
import numpy as np
import torch


import os
os.chdir('..')

from Centralised_Model.centralised_model import  Net, Loader, test, Main

def test_compare():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Run bashfile
    subprocess.call('fl_network/run_networks.sh')

    # Get FL weights from final round
    fl_weights_file = np.load('fl_network/server/round-3-weights.npz')
    fl_weights_args = fl_weights_file.files
    fl_weights = []
    for arg in fl_weights_args:
        fl_weights.append(fl_weights_file[arg])
    
    # Get centralised model weights
    aggr_dataset_path = 'Centralised_Model/Data/patient.csv'
    cent_weights = Main(epochs=10, batch_size=32, data_path=aggr_dataset_path)
    
    # Create models for testing
    test_dataset_path = 'fl_network/data_scientist/data/patient.csv'
    Model = Loader(test_dataset_path)
    testset = Model.load_test_data()
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    
    Net_CL = Net(35,10,2)
    Net_CL.set_weights(cent_weights)
    cl_loss, cl_accuracy = test(Net_CL, testloader, device)
    
    Net_FL = Net(35,10,2)
    Net_FL.set_weights(fl_weights)
    fl_loss, fl_accuracy = test(Net_FL, testloader, device)
    
