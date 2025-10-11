import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
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
    
# #Fetch Environment variables (For API Key)
load_dotenv()

#Fetch Data for the model from OpenAQ
def getData(city, pollutant = 'pm25', date_from = '2024-01-01', date_to = '2024-12-31', save_to_csv = True):
    '''
    Args:
        city (str): City name (e.g. Tokyo)
        pollutant (str): Pollutant type ('pm25', 'pm10', 'no2','o3','so2','co')
        date_from (str): Start date in YYYY-MM-DD format
        date_to (str): End date in YYYY-MM-DD format
        
    Returns:
        pandas.Dataframe: Air quality data
    '''
    
    all_data = []
    page = 1
    
    #Empty Dictionary to receive outputs of the csv file
    countryIndex = {}
    country_df = pd.read_csv("D:\VSCode\Scripts\Python\AQI Predictor\data\Country Indexes for OpenAQ API - Sheet1.csv", usecols=['id', 'name'])
    for index in range (len(country_df)):
        countryIndex[country_df.iloc[index, 1]] = country_df.iloc[index, 0]
    
    #Setting up request to API
    cityIndex = {
        'Tokyo': 'Japan',
        'Berlin': 'Germany',
        'Quebec': 'Canada',
        'Warsaw': 'Poland',
        'Paris': 'France',
        'London': 'United Kingdom',
        'Delhi': 'India',
        'Beijing': 'China'
    }
    locationIndex = countryIndex + cityIndex
    url = f"https://api.openaq.org/v3/locations/{}"

    #API Key from .env file
    api_key = os.getenv('OPENAQ_API_KEY')
    if not api_key:
        print("ERROR: No API key found")
        print("Please create .env file with your OpenAQ API Key")
        return None
    
    #Add key to headers
    headers = {
        'X-API-Key': api_key
    }
    params = {
        'city': city,
        'parameter': pollutant,
        'date_from': date_from,
        'date_to': date_to,
        'limit': 1000,
        'page': page
    }

    #Debugging because I think it's infinitely looping
    maxNumPages = 100
    while page <= maxNumPages:

        
        print(f"Fetching page {page}...")
        try:
            response = requests.get(url, params = params, headers = headers)
            response.raise_for_status() #flags bad requests like 404 as an error
            data = response.json()
            
            print(f"  Response keys: {data.keys()}")
            print(f"  Results in this page: {len(data.get('results', []))}")
            
            if 'results' not in data or len(data['results']) == 0:
                print ("No results found... Stopping")
                break
            
            all_data.extend(data['results'])
            print(f"  Total collected so far: {len(all_data)}")
            
            if 'meta' in data:
                print(f"  Meta info: {data['meta']}")
                
                if 'found' in data['meta']:
                    if len(all_data) >= data['meta']['found']:
                        print("All available data found... Stopping")
                        break
            
            
            page += 1
            
            
            #Don't overwhelm API with requests
            import time
            time.sleep(0.5)
        except requests.exceptions.HTTPError as e:
            print(f"HTTPError: {e}")
            break

        except Exception as e:
            print(f"Error: {e}")
            break 

    if all_data:
        df = pd.DataFrame(all_data)
        print(f"Total measurements fetched: {len(df)}")
        
        #save data to csv
        if save_to_csv:
            filename = f"D:\VSCode\Scripts\Python\AQI Predictor\data\{city}_{pollutant}_{date_from}_to_{date_to}.csv"
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
            
        return df
    else:
        print("No data retrieved")
        return None



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
model = Network(input_size, hidden_size, num_classes)
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
    model = Network(input_size, hidden_size, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'Using device: {device}')
    
    print("Model initialized successfully.")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())} total parameters")
    
    print("Attempting to retrieve data.")
    tokyo_data = getData('Beijing')