import torch
import torch.nn as nn

"""
File containing the base LSTM model to be used for training and predictions
"""
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

#           Program
if __name__ == "__main__":
    model = LSTM (num_inputs = 5, hidden_size = 64, num_layers = 4, output_size = 1, dropout = 0.2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(model)