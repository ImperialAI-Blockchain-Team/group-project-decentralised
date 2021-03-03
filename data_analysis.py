import pandas as pd
import numpy as np


data_na = pd.read_csv('Data/patient.csv')
labels = ['patientunitstayid','patienthealthsystemstayid','wardid','hospitaladmittime24',
          'hospitaladmitoffset', 'hospitaldischargeyear','hospitaldischargetime24',
          'hospitaldischargeoffset','unitadmittime24','unitvisitnumber','unitdischargetime24',
          'unitdischargeoffset','uniquepid','apacheadmissiondx','hospitaldischargelocation',
          'hospitaladmitsource','unitstaytype','unitdischargelocation','unitdischargestatus']
data_na = data_na.drop(columns=labels)


data = data_na.dropna(axis=0)

# Drop rows where age not a unique number
data = data.drop(data[data.age == '> 89'].index)

# Turn binary string to bool
data = data.drop(data[data.gender == 'Unknown'].index) # drop unknown gender
data = data.drop(data[data.gender == 'Other'].index)  # drop other gender
Gender = {'Male': 0,'Female': 1}
data.gender = [Gender[item] for item in data.gender]
Discharge_Status = {'Alive': 1, 'Expired':0}
data.hospitaldischargestatus = [Discharge_Status[item] for item in data.hospitaldischargestatus]
#data.unitdischargestatus = [Discharge_Status[item] for item in data.unitdischargestatus]
data.age = [int(item) for item in data.age]

# One-hot encooding
data = pd.concat([data,pd.get_dummies(data['ethnicity'], prefix='eth')],axis=1)
data.drop(['ethnicity'],axis=1, inplace=True)
data = pd.concat([data,pd.get_dummies(data['unittype'], prefix='unittype')],axis=1)
data.drop(['unittype'],axis=1, inplace=True)
data = pd.concat([data,pd.get_dummies(data['unitadmitsource'], prefix='unitadmitsource')],axis=1)
data.drop(['unitadmitsource'],axis=1, inplace=True)



# split data into 4 different sets basedd on hospital id
# Get data for hospital 1, most frequent hospital
most_frequent_1 = data['hospitalid'].mode()[0]
data_1 = data.drop(data[data.hospitalid != most_frequent_1].index)
data_1 = data_1.drop(columns=['hospitalid'])
data_1_labels = data_1['hospitaldischargestatus']
data_1_inputs = data_1.drop(columns=['hospitaldischargestatus'])
data_remainder_1 = data.drop(data[data.hospitalid == most_frequent_1].index)

# Get data for hospital 2, second most frequent hospital
most_frequent_2 = data_remainder_1['hospitalid'].mode()[0]
data_2 = data.drop(data[data.hospitalid != most_frequent_2].index)
data_2 = data_2.drop(columns=['hospitalid'])
data_2_labels = data_2['hospitaldischargestatus']
data_2_inputs = data_2.drop(columns=['hospitaldischargestatus'])
data_remainder_2 = data_remainder_1.drop(data[data.hospitalid == most_frequent_2].index)

# Get data for hospital 3, third most frequent hospital
most_frequent_3 = data_remainder_2['hospitalid'].mode()[0]
data_3 = data.drop(data[data.hospitalid != most_frequent_3].index)
data_3 = data_3.drop(columns=['hospitalid'])
data_3_labels = data_3['hospitaldischargestatus']
data_3_inputs = data_3.drop(columns=['hospitaldischargestatus'])
data_remainder_3 = data_remainder_2.drop(data[data.hospitalid == most_frequent_3].index)

# Get data for data scientist, randomly chosen hospital id excluding ones above
data_0 = data.drop(data[data.hospitalid != 272].index)
data_0 = data_0.drop(columns=['hospitalid'])
data_0_labels = data_0['hospitaldischargestatus']
data_0_inputs = data_0.drop(columns=['hospitaldischargestatus'])

data_aggregate = pd.concat((data_1,data_2,data_3),axis=0)

save_files = True
if save_files:
    data_0.to_csv('fl_network/data_scientist/data/patient.csv')
    data_1.to_csv('fl_network/client_1/data/patient.csv')
    data_2.to_csv('fl_network/client_2/data/patient.csv')
    data_3.to_csv('fl_network/client_3/data/patient.csv')
    data_aggregate.to_csv('Centralised_Model/Data/patient.csv')