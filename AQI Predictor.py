import torch
from re import A
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import requests #grab from OpenAQ
import matplotlib #Visualize Outputs
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, date #for prediction lengths


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
    
#Date Calculation for later .csv saving and predictions
currentDate = date.today()
futureDate = currentDate
def calculateTime(yearsToAdd = 0, monthsToAdd = 0, daysToAdd = 7):
    global currentDate, futureDate
    #Define appropriate variables
    isLeapYear = False
    currentYear = currentDate.year
    currentMonth = currentDate.month
    currentDay = currentDate.day
    numDaysPerMonth = {
        4 or 6 or 9 or 11:
            30,
        1 or 3 or 5 or 7 or 8 or 10 or 12:
            31,
        2:
            28
    }
    if currentYear % 4 == 0:
        isLeapYear = True
#Fetch Data for the model from OpenAQ
def getData(city, pollutant = 'pm25', date_from = currentDate, date_to = futureDate):
    url = "https://api.openaq.org/v2/measurements"
    
#           Training Parameters
input_size = 5 #Placeholder as I don't have the data yet
hidden_size = 64 #Placeholder as I don't have the data yet
num_classes = 1 #Placeholder as I don't have the data yet
num_epochs = 100
learning_rate = 0.001
batch_size = 64
num_samples = 1000

#           Initialize model, loss function, and optimizer
model = Network(input_size, hidden_size, num_classes)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


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


if __name__ == "__main__":
    #Test for CUDA
    model = Network(nn.Module())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'Using device: {device}')
    print("\n\n")
    #Test Datetime.date() return for futureDate value
    print(currentDate)
    
    print(futureDate)