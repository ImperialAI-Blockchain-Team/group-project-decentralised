#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:10:41 2021

@author: marwanmousa
"""
import pandas as pd
import numpy as np

data_na = pd.read_csv('patient.csv')
labels = ['patientunitstayid','patienthealthsystemstayid','wardid','hospitaladmittime24',
          'hospitaladmitoffset', 'hospitaldischargeyear','hospitaldischargetime24',
          'hospitaldischargeoffset','unitadmittime24','unitvisitnumber','unitdischargetime24',
          'unitdischargeoffset','uniquepid','apacheadmissiondx','hospitaldischargelocation',
          'hospitaladmitsource','unitstaytype','unitdischargelocation','unitdischargestatus']
data_na = data_na.drop(columns=labels)


data = data_na.dropna(axis=0)

data = data.drop(data[data.age == '> 89'].index)

# Turn binary string to bool
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

data_1 = data.drop(data[data.hospitalid > 100].index)
data_1 = data_1.drop(columns=['hospitalid'])

data_2 = data.drop(data[data.hospitalid <= 100].index)
data_2 = data_2.drop(data_2[data_2.hospitalid > 200].index)
data_2 = data_2.drop(columns=['hospitalid'])

data_3 = data.drop(data[data.hospitalid <= 200].index)
data_3 = data_3.drop(data_3[data_3.hospitalid > 300].index)
data_3 = data_3.drop(columns=['hospitalid'])

data_4 = data.drop(data[data.hospitalid <= 300].index)
data_4 = data_4.drop(columns=['hospitalid'])