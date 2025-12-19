import torch
import lstm_model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
from consolidate_data import consolidateData
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt #Visualize Outputs
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def splitTimeSeries(df, train_ratio = 0.7, val_ratio = 0.15):
    """
    Split the Time Series
    
    Args:
        df: DataFrame containing Historical AQI Data (Chronologically)
        train_ratio: How much of the total data set will be for training (0.7 = 70%)
        val_ratio: How much of the total data set will be for validation (0.15 = 15%)
        
    Returns:
        train_df, val_df, test_df
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    print(f"Training set:   {len(train_df)} entries ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
    print(f"Validation set:   {len(val_df)} entries ({val_df['datetime'].min()} to {val_df['datetime'].max()})")
    print(f"Testing set:   {len(test_df)} entries ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
    
    return train_df, val_df, test_df

def prepSequences (df, sequence_length = 7):
    """
    Create sequences from city data

    Args:
        df: Dataframe with features (sorted Chronologically)
        sequence_length: Number of entries per sequence (7 in this case)
    
    Returns:
        X: sequences (samples, sequence_length, features)
        y: targets (samples)
    """
    
    #       Select features
    feature_cols = ['datetime', 'parameter', 'value']
    
    #       Extract to numpy array
    data = df[feature_cols].values
    
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # Past 7 entries
        X.append(data[i:i + sequence_length])
        # Target: AQI value at the next time step
        y.append(data[i + sequence_length, 0]) #mean of pm2.5 is the first column
        
        return np.array(X), np.array(y)
    
def createLoaders(city = 'Tokyo', sequence_length = 7, batch_size = 32):
    """
    The complete process: Load Data -> Split into appropriate sets -> create sequences -> create DataLoaders
    
    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    
    #       Load the Data
    cityData = consolidateData(city)
    
    indices = []
    for index in cityData.index:
        if cityData.at[index, 'parameter'] == 'pm25':
            indices.append(index)
    
    print(f"\n{'='*60}")
    print(f"Preparing data for {city}")
    print(f"{'='*60}\n")
    
    print(f"Found {len(indices)} pm2.5 entries out of {len(cityData)} total entries for {city}")
    print(f"The acknowledged entries account for approximately {round(len(indices) / len(cityData))} % of total entries.")
    data = cityData.loc[indices].reset_index(drop=True)
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['day'] = data['datetime'].dt.day
    data['day_of_week'] = data['datetime'].dt.dayofweek
    
    #       Split into appropriate sets
    train_df, val_df, test_df = splitTimeSeries(data)
    
    #       Drop datetime column
    for d in [train_df, val_df, test_df]:
        d.drop(columns = ['datetime', 'parameter'], inplace = True, errors = 'ignore')
    #       Normalize features (Will only be applied to training data)
    feature_cols = ['value', 'year', 'month', 'day', 'day_of_week']
    
    scaler = StandardScaler()
    
    #       Fit scaler on training data
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    
    #       Transform Validation and Test sets on same scaler
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    #       Create sequences for Training, Validation, and Testing
    X_train, y_train = prepSequences(train_df, sequence_length)
    X_val, y_val = prepSequences(val_df, sequence_length)
    X_test, y_test = prepSequences(test_df, sequence_length)
    
    #       Print out the shapes for each sequence
    print(f"\nSequence Shapes")
    print(f"    Train: X={X_train.shape}, y={y_train.shape}")
    print(f"    Validation: X={X_val.shape}, y= {y_val.shape}")
    print(f"    Testing: X={X_test.shape}, y={y_test.shape}")
    
    #       Convert everything to PyTorch Tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val).unsqueeze(1)
    
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
    #       Create the Data Loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    #return Dataloaders and scaler
    return train_dataloader, val_dataloader, test_dataloader, scaler

def plotTrainingHistory(history):
    #           Plot Training History
    plt.figure(figsize=(10,6))
    plt.plot(history['train_loss'], label = 'Training Loss', linewidth = 2)
    plt.plot(history['val_loss'], label = 'Validation Loss', linewidth = 2)
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Loss (MSE)', fontsize = 12)
    plt.title('Training History', fontsize = 14, fontweight = 'bold')
    plt.legend(fontsize = 11)
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi = 300)
    plt.show()
    
    print("\nTraining History saved as 'training_history.png'")
    
def trainModel(model, criterion, optimizer, train_loader, val_loader, num_epochs = 100, learning_rate = 0.001):
    """Training the LSTM model with early stopping

    Args:
        model (LSTM): A network from the LSTM class
        criterion (nn.MSELoss): criterion for the loss function (I'm using MSE)
        optimizer (optim.Adam): Optimizer for the model (I'm using Adam)
        train_loader (_type_): DataLoader with the Training set
        val_loader (_type_): DataLoader with the Validation set
        num_epochs (int, optional): Max number of iterations to go through for training. Default is 100 for testing purposes.
        learning_rate (float, optional): The learning rate for the optimizer. Default is 0.001.
        
    Returns:
        model: The Trained LSTM model
        history: A Dictionary containing all of the losses during training for plotting purposes
    """
    #       Number of allowed Epochs with no improvement in Validation Loss
    patience = 10
    #         Cast model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model.to(device)
    
    #track learning History
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"\n{'='*60}")
    print(f"training on {device}")
    print(f"{'='*60}\n")
    
    #       Training phase
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            #       send data to GPU
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            #       forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            #       backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)
        
        #       Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)
        
        #       Save Losses to History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        #       Display progress
        print(f"Epoch [{epoch+1} / {num_epochs}]"
              f"Train Loss: {avg_train_loss:.4f}"
              f"Val Loss: {avg_val_loss:.4f}")
        
        #       Check for Stop
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_tracker = 0
            best_model_state = model.state_dict().copy()
            
            print(f"    New best validation loss!")
        else:
            patience_tracker += 1
            print(f"    No improvement ({patience_tracker} / {patience})")
            
            if patience_tracker >= patience:
                print(f"\n{'='*60}")
                print(f"Early stop triggered after {epoch+1} epochs")
                print(f"Best validation loss: {best_val_loss:.4f}")
                print(f"{'='*60}\n")
                break
    #       Load current best model
    model.load_state_dict(best_model_state)
    
    #       Return best model and training History
    return model, history

def evaluateModel(model, test_loader, scaler):
    """
    Evaluate Model effectiveness on the Test Set
    
    Args:
        model: trained LSTM model
        test_loader: DataLoader for the test data set
        scaler: StandardScaler used for normalization
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_X)
            
            all_predictions.append(predictions.cpu().numpy())
            all_actuals.append(batch_y.cpu().numpy())
    
    #       Concat all batches
    predictions = np.concatenate(all_predictions)
    actuals = np.concatenate(all_actuals)
    
    #       Denormalize predictions (convert back to original scale)
    #       Assumes PM2.5 was the first feature normalized
    def denormalize(predictions):
        dummy = np.zeros((len(predictions), 5))
        dummy[:,0] = predictions.flatten()
        return scaler.inverse_transform(dummy)[:,0]
    predictions_denorm = denormalize(predictions)
    actuals_denorm = denormalize(actuals)
    
    #       Calculate evaluation metrics
    mae = mean_absolute_error(actuals_denorm, predictions_denorm)
    rmse = np.sqrt(mean_squared_error(actuals_denorm, predictions_denorm))
    r2 = r2_score(actuals_denorm, predictions_denorm)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'predictions': predictions_denorm,
        'actuals': actuals_denorm
    }
    
    #       Print out Metrics
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION")
    print(f"{'='*60}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} µm/m³")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} µm/m³")
    print(f"R² (R2): {r2:.3f}")
    print(f"{'='*60}\n")
    
    return metrics

def plotPredictions(metrics, num_samples = 100):
    """
    Plot Predicted vs Actual values
    """
    actuals = metrics['actuals'][:num_samples]
    predictions = metrics['predictions'][:num_samples]
    
    plt.figure(figsize = (12, 6))
    
    plt.subplot(1,2,1)
    plt.plot(actuals, label = 'Actual', color = 'blue', linewidth = 2, alpha = 0.7)
    plt.plot(predictions, label = 'Predicted', color = 'red', linewidth = 2, alpha = 0.7)
    plt.xlabel('Day', fontsize = 12)
    plt.ylabel('PM2.5 Value (µm/m³)', fontsize = 12)
    plt.title('Predicted vs Actual PM2.5 Values', fontsize = 14, fontweight = 'bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha = 0.3)
    
    plt.subplot(1,2,2)
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()],
             [actuals.min(), actuals.max()], 'r--', linewidth = 2, label = 'Perfect Prediction'
    )
    plt.xlabel('Actual PM2.5 Value (µm/m³)', fontsize = 12)
    plt.ylabel('Predicted PM2.5 Value (µm/m³)', fontsize = 12)
    plt.title('Scatter Plot of Predicted vs Actual PM2.5 Values', fontsize = 14, fontweight = 'bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha = 0.3)
    
    plt.tight_layout()
    plt.savefig('predictions_plot.png', dpi = 300)
    plt.show()
    
    print("\nPredictions plot saved as 'predictions_plot.png'")
    
def main():
    #       Define Hyperparameters
    num_inputs = 5 # (value, year, month, day, day_of_week) 
    hidden_size = 64 
    output_size = 1 # Predicted PM2.5 value
    seq_length = 7
    num_layers = 2 # Number of LSTM layers
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 64
    num_samples = 1000
    num_workers = 4

    #       Prepare and Load Data
    train_loader, val_loader, test_loader, scaler = createLoaders(
        city = 'Tokyo',
        sequence_length = 7,
        batch_size = 32
        )
    
    #       Initialize Model, Loss Function, and Optimizer
    model = lstm_model.LSTM(
        input_size = num_inputs,
        hidden_size = hidden_size,
        num_layers = num_layers,
        output_size = output_size
    )
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
    
    #       Begin Training Process
    trained_model, history = trainModel(
        model = model,
        criterion = nn.MSELoss(),
        optimizer = optim.Adam(model.parameters(), lr = learning_rate),
        train_loader = train_loader,
        val_loader = val_loader,
        num_epochs = num_epochs,
        learning_rate = learning_rate
    )
    
    #       Plot Training History
    plotTrainingHistory(history)
    
    #       Evaluate Model on Test Set
    metrics = evaluateModel(trained_model, test_loader, scaler)
    
    #       Plot Predictions
    plotPredictions(metrics, num_samples)
    
    #       Save Model
    torch.save({
               'model_state_dict': trained_model.state_dict(),
               'scaler': scaler,
               'metrics': metrics,
               'history': history
               }), 'tokyo_model.pth'
    
    print("\nModel saved as 'tokyo_model.pth")
    
    return trained_model, metrics

if __name__ == "__main__":
    model, metrics = main()