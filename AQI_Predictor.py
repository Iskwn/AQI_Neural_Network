import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib #Visualize Outputs
from tqdm import tqdm #Progress Bar (for training loops and data things)

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
            
#Fetch Data from Data folder for consolidation
def consolidateData(city):
    '''
    Args:
        city (str): City name (e.g. Tokyo)
        
    Returns:
        pandas.Dataframe: Air quality data for the specified city
    '''
    city_path = Path(f"D:/VSCode/Scripts/Python/AQIPredictor/data/{city}")
    city_data = pd.DataFrame()
    sensors = sorted([d for d in city_path.iterdir() if d.is_dir()])
    print(f"Found {len(sensors)} sensors." if len(sensors) != 1 else "Found 1 sensor.")
    print("Beginning Data Collection...\n")
    #Iterates through each Sensor
    for sensor_path in tqdm(sensors, desc = f"Iterating through sensors in {city}", leave = True):
        sensor_path: Path
        sensor_name = sensor_path.name
        sensor_id = sensor_name.replace("Sensor_", "")
        sensor_data = pd.DataFrame()
        #saves years into a list
        years = sorted([x.name for x in sensor_path.iterdir() if x.is_dir()])
        print(f"Found data for {', '.join(years)} for sensor {sensor_id}")
        print(f"SENSOR {sensor_id}")
        
        #Iterates through each Year for the Sensor
        for year in tqdm(years, desc= f"Iterating through years in sensor", leave = False):
            year_path = sensor_path.joinpath(year)
            csv_files = [x.name for x in year_path.iterdir()]

            for file in tqdm(csv_files, desc= f"Adding data to {year} DataFrame", leave = False):
                file_path = year_path.joinpath(file)
                df = pd.read_csv(file_path, compression = 'gzip')
                sensor_data = pd.concat([sensor_data, df], ignore_index = True)
        city_data = pd.concat([city_data, sensor_data], ignore_index= True)
    print(f"Finished Data Collection for {city}\n")
    print(f"Total files collected: {len(city_data)}")
    return city_data
        
    

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



# #           Initialize model, loss function, and optimizer
# model = LSTM(num_inputs, hidden_size)
# criterion = nn.MSELoss() 
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# #           Load and Preprocess Data
# with torch.no_grad():
#     #Placeholders
#     data = pd.DataFrame(np.random.rand(num_samples,6)) # 5 features + 1 target
#     data = data.dropna()
    
#     #normalize data
#     data = (data- data.mean()) / data.std()
    


# #           Create Dataset and DataLoader
# dataset_array = data.to_numpy()
# X = torch.tensor(dataset_array[:, :-1], dtype = torch.float32) #Features
# y = torch.tensor(dataset_array[:, -1], dtype = torch.float32).unsqueeze(1) #Fixes shape of the Target

# dataset = TensorDataset(X, y)



#           Program
if __name__ == "__main__":
    #Test for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = LSTM(num_inputs, hidden_size, num_layers, output_size).to(device)
    print("Model initialized successfully.")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())} total parameters")
    
    print("\nReady for training on test data.")
    print("Beginning forward pass test...")
    
    x = torch.randn(batch_size, seq_length, num_inputs).to(device)
    output: torch.Tensor = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr= learning_rate)
    target = torch.randn(batch_size, 1).to(device)

    loss = criterion(output, target)
    print("Forward and Backward pass complete successfully.")
    print(f"Test Loss: {loss.item():.4f}")
    
    print("\nModel Architecture:")
    print(model)