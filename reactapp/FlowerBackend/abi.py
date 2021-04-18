job_abi = [
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "_contractAddressDatasets",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "_contractAddressModels",
          "type": "address"
        }
      ],
      "payable": False,
      "stateMutability": "nonpayable",
      "type": "constructor"
    },
    {
      "constant": True,
      "inputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "name": "jobs",
      "outputs": [
        {
          "internalType": "address payable",
          "name": "owner",
          "type": "address"
        },
        {
          "internalType": "string",
          "name": "modelIpfsHash",
          "type": "string"
        },
        {
          "internalType": "string",
          "name": "strategyHash",
          "type": "string"
        },
        {
          "internalType": "uint256",
          "name": "minClients",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "initTime",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "daysUntilStart",
          "type": "uint256"
        },
        {
          "internalType": "bool",
          "name": "active",
          "type": "bool"
        },
        {
          "internalType": "bool",
          "name": "registered",
          "type": "bool"
        },
        {
          "internalType": "bool",
          "name": "trainingStarted",
          "type": "bool"
        },
        {
          "internalType": "uint256",
          "name": "holdingFee",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "numAllow",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "bounty",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "feeSum",
          "type": "uint256"
        }
      ],
      "payable": False,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": True,
      "inputs": [
        {
          "internalType": "string",
          "name": "_modelIpfsHash",
          "type": "string"
        }
      ],
      "name": "isSenderModelOwner",
      "outputs": [
        {
          "internalType": "bool",
          "name": "",
          "type": "bool"
        }
      ],
      "payable": False,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": True,
      "inputs": [
        {
          "internalType": "string",
          "name": "_modelIpfsHash",
          "type": "string"
        }
      ],
      "name": "isSenderDatasetOwner",
      "outputs": [
        {
          "internalType": "bool",
          "name": "",
          "type": "bool"
        }
      ],
      "payable": False,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": True,
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_id",
          "type": "uint256"
        }
      ],
      "name": "getJobStartTime",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "payable": False,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": True,
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_id",
          "type": "uint256"
        }
      ],
      "name": "registrationPeriodOver",
      "outputs": [
        {
          "internalType": "bool",
          "name": "",
          "type": "bool"
        }
      ],
      "payable": False,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": False,
      "inputs": [
        {
          "internalType": "string",
          "name": "_modelIpfsHash",
          "type": "string"
        },
        {
          "internalType": "string",
          "name": "_strategyHash",
          "type": "string"
        },
        {
          "internalType": "uint256",
          "name": "_minClients",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "_daysUntilStart",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "_bounty",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "_holdingFee",
          "type": "uint256"
        }
      ],
      "name": "createJob",
      "outputs": [],
      "payable": True,
      "stateMutability": "payable",
      "type": "function"
    },
    {
      "constant": False,
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_id",
          "type": "uint256"
        },
        {
          "internalType": "string",
          "name": "_datasetIpfsHash",
          "type": "string"
        }
      ],
      "name": "registerDatasetOwner",
      "outputs": [],
      "payable": True,
      "stateMutability": "payable",
      "type": "function"
    },
    {
      "constant": False,
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_id",
          "type": "uint256"
        },
        {
          "internalType": "address payable",
          "name": "_datasetOwner",
          "type": "address"
        }
      ],
      "name": "addToAllowList",
      "outputs": [],
      "payable": False,
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "constant": True,
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_id",
          "type": "uint256"
        }
      ],
      "name": "getNumRegistered",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "payable": False,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": True,
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_id",
          "type": "uint256"
        }
      ],
      "name": "getNumAllow",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "payable": False,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": True,
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_id",
          "type": "uint256"
        }
      ],
      "name": "getJobDetails",
      "outputs": [
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        },
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "payable": False,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": True,
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_id",
          "type": "uint256"
        }
      ],
      "name": "getJobStatus",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        },
        {
          "internalType": "bool",
          "name": "",
          "type": "bool"
        },
        {
          "internalType": "bool",
          "name": "",
          "type": "bool"
        }
      ],
      "payable": False,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": False,
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_id",
          "type": "uint256"
        }
      ],
      "name": "start_job",
      "outputs": [],
      "payable": False,
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "constant": False,
      "inputs": [
        {
          "internalType": "uint256",
          "name": "_id",
          "type": "uint256"
        }
      ],
      "name": "withdrawFee",
      "outputs": [],
      "payable": False,
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "constant": False,
      "inputs": [
        {
          "internalType": "uint256",
          "name": "jobId",
          "type": "uint256"
        }
      ],
      "name": "deactivateJob",
      "outputs": [],
      "payable": False,
      "stateMutability": "nonpayable",
      "type": "function"
    }
]