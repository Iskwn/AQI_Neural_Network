import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib #Visualize Outputs
from consolidate_data import consolidateData

#           Define LSTM Neural Network
class LSTM(nn.Module):
    #           class constructor   
    def __init__(self, num_inputs = 5, hidden_size = 64, num_layers = 2, output_size = 1, dropout=0.2):
        """
        LSTM Neural Network for AQI Prediction
        
        Args:
            input_size: Number of features per timestep (e.g., 4: pm2.5, month, day_of_week, day_of_year)
            hidden_size: Size of the LSTM hidden state (e.g. 64 or 128)
            num_layers: Number of stacked LSTM layers (e.g. 2 or 3)
            output_size: Number of output features (e.g., 1 for next day pm2.5 prediction)
            dropout: Dropout probability (applied between layers)
        """        
        super(LSTM, self).__init__()
           
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #LSTM layer
        self.lstm = nn.LSTM(
            input_size = num_inputs, #Number of features at timestep (day)
            hidden_size = hidden_size, # Size of the hidden state
            num_layers = num_layers, # Number of stacked LSTM layers
            batch_first = True, # Input and output tensors are provided as (batch, seq, feature)
            dropout = dropout if num_layers > 1 else 0 # Dropout between LSTM layers
        )   
        #           fully connected layer to make final prediction
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        #Iniitialize Hidden and Cell States
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        #Forward Propagate the Network
        # out shape: (batch_size, seq_length, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        #Uses last time step's output
        out = out[:, -1, :]
        
        #Pass through fc layer
        out = self.fc(out)
        
        #Return Output
        return out
            
#           Training Parameters
num_inputs = 5 
hidden_size = 64 
output_size = 1
seq_length = 20
num_layers = 2
num_epochs = 100
learning_rate = 0.001
batch_size = 64
num_samples = 1000
num_workers = 4



#           Initialize model, loss function, and optimizer
model = LSTM(num_inputs, hidden_size)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#           Potential Cities for Prediction
cities = ['Tokyo', 'Quebec', 'Berlin', 'London', 'Beijing', 'Delhi', 'Warsaw', 'Paris']
#           Program
if __name__ == "__main__":
    #Test for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    #       Get City for Data Consolidation
    print("Select City you wish to Predict the AQI for:\n")
    for i, city in enumerate(cities):
        print(f"{i+1}. {city}")
    print("0. Exit Program\n")
    
    city = int(input("Enter your choice: "))
    
    if city == 0:
        print("\nExiting Program.")
        sys.exit()
        
    print(f"\nYou selected {cities[city-1]}\n")
    print(f"\nBeginning Data Consolidation for {cities[city-1]}...\n")
    cityData = consolidateData(cities[city-1])
    
    print(cityData.head())
    
    print("\nData Consolidation Complete. Proceeding to Model Training...\n")
    
    #           Prepare Data for Training
    pm25_indices = []
    for index in cityData.index:
        if cityData.at[index, 'parameter'] == 'pm25':
            pm25_indices.append(index)
            
    print(f"Found {len(pm25_indices)} pm2.5 entries for {cities[city-1]} out of {len(cityData)} total entries. Accounting for approximately {round(len(pm25_indices) / len(cityData) * 100, 5)}%\n")
    pm25_data = cityData.loc[pm25_indices].reset_index(drop=True)
    print(pm25_data.head())