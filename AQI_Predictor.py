import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib #Visualize Outputs


#           Define the Neural Network
class basicNetwork(nn.Module):
    #           class constructor
    def __init__(self, num_inputs = 5, hidden_size = 64, num_classes = 1):
        super(basicNetwork, self).__init__() #makes sure __init__ is called correctly
        
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
    
#Fetch Data from Data folder
def getData(city, pollutant = 'pm25'):
    '''
    Args:
        city (str): City name (e.g. Tokyo)
        pollutant (str): Pollutant type ('pm25', 'pm10', 'no2','o3','so2','co')
        save_to_csv (bool): Whether or not the pandas df is saved as a csv file
        
    Returns:
        pandas.Dataframe: Air quality data
    '''
    city_path = Path(f"D:/VSCode/Scripts/Python/AQIPredictor/data/{city}")
    sensors = sorted([d for d in city_path.iterdir() if d.is_dir()])
    print(sensors)
    
    print(f"Found {len(sensors)} sensors.")
    
    #Iterates through each Sensor
    for sensor_path in sensors:
        sensor_name = sensor_path.name
        sensor_id = sensor_name.replace("Sensor_", "")
        
        #saves years into a list
        years = sorted([x.name for x in sensor_path.iterdir() if x.is_dir()])
        
        print(f"Found data for {', '.join(years)} for sensor {sensor_id}")
        print(f"SENSOR {sensor_id}")
        for year in sensor_path.iterdir():
            print(f"Iterating through {year.name}")
            year_path = sensor_path.joinpath(year)
            sensor_data = [x.name for x in year_path.iterdir()]
            print(f"\nfound {len(sensor_data)} items\n")
            print(f"Finished Iterating through {year.name}\n")
    
    

#           Training Parameters
input_size = 5 #Placeholder as I don't have the data yet
hidden_size = 64 #Placeholder as I don't have the data yet
num_classes = 1 #Placeholder as I don't have the data yet
num_epochs = 100
learning_rate = 0.001
batch_size = 64
num_samples = 1000
num_workers = 4



#           Initialize model, loss function, and optimizer
model = basicNetwork(input_size, hidden_size, num_classes)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



#           Load and Preprocess Data
with torch.no_grad():
    #Placeholders
    data = pd.DataFrame(np.random.rand(num_samples,6)) # 5 features + 1 target
    data = data.dropna()
    
    #normalize data
    data = (data- data.mean()) / data.std()
    


#           Create Dataset and DataLoader
dataset_array = data.to_numpy()
X = torch.tensor(dataset_array[:, :-1], dtype = torch.float32) #Features
y = torch.tensor(dataset_array[:, -1], dtype = torch.float32).unsqueeze(1) #Fixes shape of the Target

dataset = TensorDataset(X, y)



#           Program
if __name__ == "__main__":
    #Test for CUDA
    model = basicNetwork(input_size, hidden_size, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'Using device: {device}')
    
    print("Model initialized successfully.")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())} total parameters")
    
    print("Attempting to retrieve data.")
    tokyo_data = getData('Tokyo')