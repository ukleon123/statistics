import pandas as pd
import numpy as np

def split(data, name):
    result = []
    result.append(data[['SepalLengthCm']].rename(columns = {'SepalLengthCm':name}).reset_index(drop = True))
    result.append(data[['SepalWidthCm']].rename(columns = {'SepalWidthCm':name}).reset_index(drop = True))
    result.append(data[['PetalLengthCm']].rename(columns = {'PetalLengthCm':name}).reset_index(drop = True))
    result.append(data[['PetalWidthCm']].rename(columns = {'PetalWidthCm':name}).reset_index(drop = True))
    
    return result


data = pd.read_csv('./Iris.csv', sep = ',')

setosa = data[:50]
versicolor = data[50:100]
virginica = data[100:]

setosa = split(setosa, 'setosa')
versicolor = split(versicolor, 'versicolor')
virginica = split(virginica, 'virginica')

grand_mean = []
group_mean = []
features = []

for idx in range(4):
    partial = pd.concat([setosa[idx], 
                      versicolor[idx], 
                      virginica[idx]], axis = 1)
    group_mean.append(partial.sum(axis = 0) / 50)
    grand_mean.append(partial.sum(axis = 1).sum(axis = 0) / 150)
    features.append(partial)
SSB = [0, 0, 0, 0]
SSE = [0, 0, 0, 0]

for idx in range(4):
    SSE[idx] = SSE[idx] + (setosa[idx].sub(group_mean[idx].iat[0])['setosa'] ** 2).sum(axis = 0)
    SSE[idx] = SSE[idx] + (versicolor[idx].sub(group_mean[idx].iat[1])['versicolor'] ** 2).sum(axis = 0)
    SSE[idx] = SSE[idx] + (virginica[idx].sub(group_mean[idx].iat[2])['virginica'] ** 2).sum(axis = 0)
    SSB[idx] =  50 * (((group_mean[idx].sub(grand_mean[idx]))) ** 2).sum(axis = 0)


for i in range(4):
    print((SSB[i] / 2) / (SSE[i] / (3 * 49)))
    

