import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import requests #grab from OpenAQ
import matplotlib #Visualize Outputs
from torch.utils.data import DataLoader, TensorDataset


#           Define the Neural Network
class Network(nn.Module):
    #           class constructor
    def __init__(self, num_inputs = 5, hidden_size = 64, num_classes = 1):
        super(Network, self).__init__() #makes sure __init__ is called correctly
        
        #           Define layers (adjust as needed)
        self.fc1 = nn.Linear(5, hidden_size) #Layer 1
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Layer 2
        self.fc3 = nn.Linear(hidden_size, num_classes) #Output Layer
        
        #           Activation and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    #           forward pass
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    

    def getData(city, pollutant, date_from, date_to):
        url = "https://openaq.org"
#           Data Parameters
input_size = 5 #Placeholder as I don't have the data yet
hidden_size = 64 #Placeholder as I don't have the data yet
num_classes = 1 #Placeholder as I don't have the data yet


#           Training Parameters
num_epochs = 100
learning_rate = 0.001

#           Initialize model, loss function, and optimizer
model = Network(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss() #Might change to MSELoss for regression if needed
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#           DataLoader Parameters
batch_size = 64
num_samples = 1000 #Placeholder as I don't have the data yet

#           Initialize placeholder data

#Load and preprocess data (without gradient calculation)
with torch.no_grad():
    #All placeholders as I don't have the data yet
    # data = pd.read_csv('AirQualityUCI.csv', sep=';') (No actual data file as of yet)
    data = pd.DataFrame(np.random.rand(num_samples, 7)) #Placeholder data
    data = data.dropna() #Drop rows with missing values
    data = data.iloc[:, 2:7] #Select relevant features
    data = (data - data.mean()) / data.std() #Normalize the data
    
#Create dataset and dataloader
dataset = pd.DataFrame(data).to_numpy()
dataset = TensorDataset(torch.tensor(dataset[:, :-1], dtype=torch.float32), torch.tensor(dataset[:, -1], dtype=torch.float32))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Using device: {device}')