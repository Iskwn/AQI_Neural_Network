import torch
import lstm_model
import torch.nn as nn
from tkinter import font
import torch.optim as optim
from matplotlib.pylab import f
from torch.utils.data import DataLoader, TensorDataset

import os
import time
import numpy as np
import pandas as pd
from consolidate_data import consolidateData
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt #Visualize Outputs
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from baseline_models import persistence_model, moving_average_model, simple_mean_model

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
    
    train_df = df[:train_end].copy()
    val_df = df[train_end:val_end].copy()
    test_df = df[val_end:].copy()
    
    if 'datetime' in train_df.columns:
        if isinstance(train_df['datetime'].dtype, pd.DatetimeTZDtype):
            #       Converts to timezone-naive if timezone-inclusive for cleaner printing
            train_df['datetime'] = train_df['datetime'].dt.tz_localize(None)
            val_df['datetime'] = val_df['datetime'].dt.tz_localize(None)
            test_df['datetime'] = test_df['datetime'].dt.tz_localize(None)
    
        print(f"Training set:   {len(train_df)} entries ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
        print(f"Validation set:   {len(val_df)} entries ({val_df['datetime'].min()} to {val_df['datetime'].max()})")
        print(f"Testing set:   {len(test_df)} entries ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
    else:
        print(f"Training set: {len(train_df)} entries")
        print(f"Validation set: {len(val_df)} entries")
        print(f"Testing set: {len(test_df)} entries")
    return train_df, val_df, test_df

def prepSequences (df, sequence_length = 24):
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
    feature_cols = ['pm25', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']
    
    #       Extract to numpy array
    data = df[feature_cols].values
    
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # Past 7 entries
        X.append(data[i:i + sequence_length])
        # Target: AQI value at the next time step
        y.append(data[i + sequence_length, 0]) #mean of pm2.5 is the first column
        
    return np.array(X), np.array(y)
    
def createLoaders(city = 'Tokyo', sequence_length = 24, batch_size = 32, num_workers = 0, return_raw = False):
    """
    The complete process: Load Data -> Split into appropriate sets -> create sequences -> create DataLoaders
    
    Returns:
        train_loader, val_loader, test_loader, scaler
        If return_raw is True, also returns raw_data_train, raw_data_test
    """
    try:
        #       Load the Data
        cityData = consolidateData(city)
        
        print(f"\n{'='*60}")
        print(f"Preparing data for {city}")
        print(f"{'='*60}\n")
        
        print(f"Initial data shape: {cityData.shape}")
        #       Filter for PM2.5 data only
        cityData = cityData[cityData['parameter'] == 'pm25'].copy()
        print(f"Found {len(cityData)} pm2.5 entries for {city} after filtering")
        
        #       Ensure datetime is properly formatted
        cityData['datetime'] = pd.to_datetime(cityData['datetime'], utc=True, errors = 'coerce')
        cityData['datetime'] = cityData['datetime'].dt.tz_localize(None) #Removes any potential timezones
        #       Drop rows with invalid datetime
        cityData = cityData.dropna(subset = ['datetime'])
        
        #       Sort by datetime
        cityData = cityData.sort_values('datetime').reset_index(drop=True)

        #       Average multiple readings per datetime (multiple sensors)
        cityData = cityData.groupby('datetime')['value'].mean().reset_index()

        #       Rename 'value' column to 'pm25' for clarity
        cityData.rename(columns = {'value': 'pm25'}, inplace = True)
        
        print(f"After aggregation: {len(cityData)} unique timestamps")
        print(f"Date range: {cityData['datetime'].min()} to {cityData['datetime'].max()}")
        
        #       Check for insufficient data
        if len(cityData) < sequence_length * 10:
            raise ValueError(f"Insufficient data for {city}: only {len(cityData)} entries found.")
        
        #       Handle missing values - fill gaps in the time series
        #       Determine frequency (assuming hourly data)
        time_diff = cityData['datetime'].diff().median()
        if time_diff < pd.Timedelta(hours = 2):
            freq = 'H'
        else:
            freq = 'D'
        
        print(f"Detected frequency: {freq}")
        
        #       Create complete date range
        date_range = pd.date_range(start = cityData['datetime'].min(), end = cityData['datetime'].max(), freq = freq)
        
        #       Reindex to fill missing timestamps
        cityData = cityData.set_index('datetime').reindex(date_range).reset_index()
        cityData.rename(columns = {'index': 'datetime'}, inplace = True)
        
        #       Interpolate missing values
        cityData['pm25'] = cityData['pm25'].interpolate(method = 'linear').bfill().ffill()
        
        #       Remove any remaining NaN (if any)
        cityData = cityData.dropna(subset = ['pm25'])
        
        print(f"After filling gaps: {len(cityData)} entries")
        
        #       Log transform to reduce skewness
        cityData['pm25'] = np.log1p(cityData['pm25'])
        
        #       Extract Date Time Features (Cyclical Encoding)
        cityData['month_sin'] = np.sin(2 * np.pi * cityData['datetime'].dt.month / 12)
        cityData['month_cos'] = np.cos(2 * np.pi * cityData['datetime'].dt.month / 12)
        cityData['day_of_week_sin'] = np.sin(2 * np.pi * cityData['datetime'].dt.dayofweek / 7)
        cityData['day_of_week_cos'] = np.cos(2 * np.pi * cityData['datetime'].dt.dayofweek / 7)
        
        #       Split into appropriate sets
        train_df, val_df, test_df = splitTimeSeries(cityData)
        
        #       Save raw data if needed
        if return_raw:
                raw_data_test = test_df.copy()
                raw_data_train = train_df.copy()
                raw_data_test = np.expm1(raw_data_test['pm25'])
                raw_data_train = np.expm1(raw_data_train['pm25'])

        #       Drop datetime column
        for d in [train_df, val_df, test_df]:
            d.drop(columns = ['datetime', 'parameter'], inplace = True, errors = 'ignore')

        #       Normalize features (Will only be applied to training data)
        feature_cols = ['pm25', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']
        
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
        print(f"    Train: X={X_train.shape}, y= {y_train.shape}")
        print(f"    Validation: X={X_val.shape}, y= {y_val.shape}")
        print(f"    Testing: X={X_test.shape}, y= {y_test.shape}")
        
        
        #       Check for potential issues
        if len(X_test) < 10:
            print(f"\n WARNING: Test set only has {len(X_test)} sequences!")
            print(f"    Consider using longer data range or different split ratios")

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
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers, pin_memory = True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = num_workers, pin_memory = True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = num_workers, pin_memory = True)
        
        #return Dataloaders and scaler
        if return_raw:
            return train_dataloader, val_dataloader, test_dataloader, scaler, raw_data_train, raw_data_test
        else:
            return train_dataloader, val_dataloader, test_dataloader, scaler
    except Exception as e:
        print(f"\nERROR preparing data for {city}:")
        print(f"    {type(e).__name__}: {e}")
        raise

def plotTrainingHistory(history, city = 'Tokyo'):
    #           Plot Training History
    plt.figure(figsize=(10,6))
    plt.plot(history['train_loss'], label = 'Training Loss', linewidth = 2)
    plt.plot(history['val_loss'], label = 'Validation Loss', linewidth = 2)
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Loss (MSE)', fontsize = 12)
    plt.title(f'Training History in {city}', fontsize = 14, fontweight = 'bold')
    plt.legend(fontsize = 11)
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(f'D:/VSCode/Scripts/Python/AQIPredictor/data/Trained Models/{city}/Graphs/{city.lower()}_training_history.png', dpi = 300)
    plt.show()
    
    print(f"\nTraining History saved as 'training_history_{city.lower()}.png'")
    
def trainModel(model, criterion, optimizer, train_loader, val_loader, num_epochs = 100, save_model_state = True):
    """Training the LSTM model with early stopping

    Args:
        model (LSTM): A network from the LSTM class
        criterion (nn.MSELoss): criterion for the loss function (I'm using MSE)
        optimizer (optim.Adam): Optimizer for the model (I'm using Adam)
        train_loader (_type_): DataLoader with the Training set
        val_loader (_type_): DataLoader with the Validation set
        num_epochs (int, optional): Max number of iterations to go through for training. Default is 100 for testing purposes.
        
    Returns:
        model: The Trained LSTM model
        history: A Dictionary containing all of the losses during training for plotting purposes
    """
    #       Number of allowed Epochs with no improvement in Validation Loss
    patience = 15
    patience_tracker = 0
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
        start_time = time.time()
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
        end_time = time.time()
        epoch_duration = end_time - start_time
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
        print(f"Epoch Duration: {epoch_duration:.2f} seconds")
        print(f"Thorughput: {len(train_loader.dataset) / epoch_duration:.0f} samples/second")
        print(f"Epoch [{epoch+1} / {num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
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

def evaluateModel(model, test_loader, scaler, save_model_state = True):
    """
    Evaluate Model effectiveness on the Test Set
    
    Args:
        model: trained LSTM model
        test_loader: DataLoader for the test data set
        scaler: StandardScaler used for normalization
        save_model_state: Boolean to save the model's state dictionary (default: True)
    
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
        rescaled_log_data = scaler.inverse_transform(dummy)[:,0]
        final_predictions = np.maximum(0, np.expm1(rescaled_log_data))
        return final_predictions
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
    
    if save_model_state:
        metrics['model_state_dict'] = model.state_dict()
    #       Print out Metrics
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION")
    print(f"{'='*60}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} µm/m³")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} µm/m³")
    print(f"R² (R2): {r2:.3f}")
    print(f"{'='*60}\n")
    
    return metrics

def evaluateWithBaselines(city = 'Tokyo', sequence_length = 24, batch_size = 32, num_workers = 0, num_epochs = 100, learning_rate = 0.001):
    """
    Train LSTM and evaluate against baseline models

    Returns:
        metrics: Dictionary of evaluation metrics for all models
        history: Training history
        city: City name
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING {city.upper()} - LSTM vs BASELINES")
    print(f"{'='*60}\n")
    
    #       Define Hyperparameters
    num_inputs = 5 # (value, year, month, day, day_of_week) 
    hidden_size = 128 
    output_size = 1 # Predicted PM2.5 value
    seq_length = 24
    num_layers = 2 # Number of LSTM layers
    batch_size = 32
    num_workers = 1
    
    if city == 'Delhi':
        batch_size = 512
        num_workers = 4
        
    #       Prepare and Load Data + Raw Data for baselines
    train_loader, val_loader, test_loader, scaler, raw_train_df, raw_test_df = createLoaders(
        city = city,
        sequence_length = seq_length,
        batch_size = batch_size,
        num_workers = num_workers,
        return_raw = True #Baseline models need raw data
        )
    
    #       Begin LSTM Training
    #       Initialize Model, Loss Function, and Optimizer
    print(f"\n{'='*60}")
    print("TRAINING LSTM MODEL")
    print(f"{'='*60}\n")
    model = lstm_model.LSTM(
        num_inputs = num_inputs,
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
        num_epochs = num_epochs
    )
    
    #       Evaluate LSTM Model
    lstm_metrics = evaluateModel(trained_model, test_loader, scaler)
    lstm_metrics['name'] = 'LSTM'
    
    #       Evaluate basline models
    print(f"\n{'='*60}")
    print("EVALUATING BASELINE MODELS")
    print(f"{'='*60}\n")
    
    #       Persistence Model
    persistence_predictions, persistence_actuals, persistence_metrics = persistence_model(raw_train_df, raw_test_df)
    print("PERSISTENCE MODEL:")
    print(f"    MAE: {persistence_metrics['MAE']:.2f} µm/m³")
    print(f"    RMSE: {persistence_metrics['RMSE']:.2f} µm/m³")
    
    #       7-Day Moving Average
    ma7_predictions, ma7_actuals, ma7_metrics = moving_average_model(raw_train_df, raw_test_df, window = 7)
    print("\n7-DAY MOVING AVERAGE MODEL:")
    print(f"    MAE: {ma7_metrics['MAE']:.2f} µm/m³")
    print(f"    RMSE: {ma7_metrics['RMSE']:.2f} µm/m³")
    
    #       30-Day Moving Average
    ma30_predictions, ma30_actuals, ma30_metrics = moving_average_model(raw_train_df, raw_test_df, window = 30)
    print("\n30-DAY MOVING AVERAGE MODEL:")
    print(f"    MAE: {ma30_metrics['MAE']:.2f} µm/m³")
    print(f"    RMSE: {ma30_metrics['RMSE']:.2f} µm/m³")
    
    #       Simple Mean Model
    smm_predictions, smm_actuals, smm_metrics = simple_mean_model(raw_train_df, raw_test_df)
    print("\nSIMPLE MEAN MODEL:")
    print(f"    MAE: {smm_metrics['MAE']:.2f} µm/m³")
    print(f"    RMSE: {smm_metrics['RMSE']:.2f} µm/m³")
    
    #       Compile metrics into a single dictionary
    metrics = {
        'LSTM' : lstm_metrics,
        'PERSISTENCE': persistence_metrics,
        'MA7' : ma7_metrics,
        'MA30' : ma30_metrics,
        'SMM' : smm_metrics
    }
    return metrics, history, city
def plotPredictions(metrics, num_samples = 100, city = 'Tokyo'):
    """
    Plot Predicted vs Actual values
    """
    actuals = metrics['LSTM']['actuals'][:num_samples]
    predictions = metrics['LSTM']['predictions'][:num_samples]
    
    plt.figure(figsize = (12, 6))
    
    plt.subplot(1,2,1)
    plt.plot(actuals, label = 'Actual', color = 'blue', linewidth = 2, alpha = 0.7)
    plt.plot(predictions, label = 'Predicted', color = 'red', linewidth = 2, alpha = 0.7)
    plt.xlabel('Hour', fontsize = 12)
    plt.ylabel('PM2.5 Value (µm/m³)', fontsize = 12)
    plt.title(f'Predicted vs Actual PM2.5 Values in {city}', fontsize = 14, fontweight = 'bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha = 0.3)
    
    plt.subplot(1,2,2)
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()],
             [actuals.min(), actuals.max()], 'r--', linewidth = 2, label = 'Perfect Prediction'
    )
    plt.xlabel('Actual PM2.5 Value (µm/m³)', fontsize = 12)
    plt.ylabel('Predicted PM2.5 Value (µm/m³)', fontsize = 12)
    plt.title(f'Scatter Plot of Predicted vs Actual PM2.5 Values in {city}', fontsize = 14, fontweight = 'bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha = 0.3)
    
    plt.tight_layout()
    plt.savefig(f'D:/VSCode/Scripts/Python/AQIPredictor/data/Trained Models/{city}/Graphs/predictions_plot_{city.lower()}.png', dpi = 300)
    plt.show()
    
    print(f"\nPredictions plot saved as 'predictions_plot_{city.lower()}.png'")
    
def main(city = 'Tokyo'):
    """
    Runs the full training and evaluation pipeline for a given city
        
    """
    #       Define Hyperparameters
    num_inputs = 5 # (value, year, month, day, day_of_week) 
    hidden_size = 128 
    output_size = 1 # Predicted PM2.5 value
    seq_length = 24
    num_layers = 2 # Number of LSTM layers
    num_epochs = 200
    learning_rate = 0.0001
    num_samples = 1000 # for plotting
    batch_size = 32
    num_workers = 1
    
    if city == 'Delhi':
        batch_size = 512
        num_workers = 4
        
    #       Train and Eval with Baselines
    metrics, history, city = evaluateWithBaselines(
        city = city,
        sequence_length = seq_length,
        batch_size = batch_size,
        num_workers = num_workers,
        num_epochs = num_epochs,
        learning_rate = learning_rate
    )
    
    #       Plot Training History (LSTM only)
    plotTrainingHistory(history, city)
    
    #       Plot LSTM Predictions
    plotPredictions(metrics, num_samples, city)
    
    #       Comparison Plots
    plotTable(metrics, city)
    plotBar(metrics, city)
    plotAllPredictions(metrics, city, num_samples)
    
    #       Residual Analysis
    plotResiduals(metrics, city, num_samples)
    
    #       Improvement Analysis (over Persistence)
    plotImprovementPercentage(metrics, city)
    
    #       Final Results Summary
    printFinalResults(metrics, city)
    
    #       Save Model
    lstm_model_data = metrics['LSTM']
    torch.save({
               'model_state_dict': lstm_model_data.get('model_state_dict'),
               'city': city,
               'metrics': metrics,
               'history': history
               }, f'D:/VSCode/Scripts/Python/AQIPredictor/data/Trained Models/{city}/{city.lower()}_model.pth')
    
    print(f"\nModel saved as '{city.lower()}_model.pth'")

def grabInfo(city = 'Tokyo', metric_name = None):
    """
    Accesses saved model data and returns specific metrics or the full dictionary

    Args:
        city (str): Name of the city (e.g., 'Tokyo')
        metric_name (str): Specific metric to pull ('MAE', 'RMSE', 'R2'). If None, returns full dictionary.
    """
    #       Construct file path
    saved_model = f'D:/VSCode/Scripts/Python/AQIPredictor/data/Trained Models/{city}/{city.lower()}_model.pth'
    
    #       Check for existence (to avoid crashes)
    if not os.path.exists(saved_model):
        print(f"Error: No saved data found for {city} at {saved_model}")
        return None
    
    try:
        #       Load the checkpoint (map_location handles loading GPU models on CPU machines)
        checkpoint = torch.load(saved_model, map_location = torch.device('cpu'))
        
        if metric_name:
            #       Look inside the 'metrics' dictionary for the metric listed
            metrics = checkpoint.get('metrics', {})
            val = metrics.get(metric_name)
            
            if val is not None:
                return val
            else:
                print(f"Metric {metric_name}' not found. Available: {list(metrics.keys())}")
                return None
        return checkpoint
    except Exception as e:
        print(f"An error occured while loading the model: {e}")
        return None

def plotTable(metrics, city = 'Tokyo'):
    fig, ax = plt.subplots(figsize = (12, 5))
    ax.axis('tight')
    ax.axis('off')
    
    #prepare data for table
    headers = ['Model', 'MAE (µm/m³)', 'RMSE (µm/m³)', 'R²', 'Improvement over Persistence MAE (%)']
    table_data = []
    
    persistence_mae = metrics['PERSISTENCE']['MAE']
    for model_name, metric in metrics.items():
        r2_str = f"{metric['R2']:.3f}" if metric['R2'] is not None else "N/A"
        
        #       Calculate Improvement over Persistence Model
        if model_name == 'Persistence':
            improvement = "-"
        else:
            improvement = f"{((persistence_mae - metric['MAE']) / persistence_mae) * 100:.2f}%"
        table_data.append([
            model_name,
            f"{metric['MAE']:.2f}",
            f"{metric['RMSE']:.2f}",
            r2_str,
            improvement
        ])
    
    #       Create Data Table
    table = ax.table(cellText = table_data, colLabels = headers, cellLoc = 'center', loc = 'center', colWidths = [0.25, 0.18, 0.18, 0.15, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    #       Style Header
    for i in range(len(headers)):
        table[(0,i)].set_facecolor('#2196F3') #Blue
        table[(0,i)].set_text_props(weight = 'bold', color = 'white')
    
    #       Highlight the best scores
    mae_values = [metric['MAE'] for metric in metrics.values()]
    rmse_values = [metric['RMSE'] for metric in metrics.values()]
    
    best_mae_idx = mae_values.index(min(mae_values))
    best_rmse_idx = rmse_values.index(min(rmse_values))
    
    table[(best_mae_idx + 1, 1)].set_facecolor('#4CAF50') #Green
    table[(best_rmse_idx + 1, 2)].set_facecolor('#4CAF50') #Green
    
    plt.title(f'Model Performance Comparison - {city}', fontsize = 14, fontweight = 'bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'D:/VSCode/Scripts/Python/AQIPredictor/data/Trained Models/{city}/Graphs/{city.lower()}_model_comparison_table.png', dpi = 300)
    plt.show()
    
    print("\nComparison table saved")
def plotBar(metrics, city = 'Tokyo'):
    models = list(metrics.keys())
    mae_values = [metrics[model]['MAE'] for model in models]
    rmse_values = [metrics[model]['RMSE'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14,7))
    
    bars1 = ax.bar(x - width/2, mae_values, width, label = 'MAE', color = '#2196F3', alpha = 0.8)
    bars2 = ax.bar(x + width/2, rmse_values, width, label='RMSE', color = '#FF9800', alpha = 0.8)
    
    ax.set_xlabel('Model', fontsize = 13, fontweight = 'bold')
    ax.set_ylabel('Error (µm/m³)', fontsize = 13, fontweight = 'bold')
    ax.set_title(f"Model Performance Comparison - {city}", fontsize = 15, fontweight = 'bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation = 20, ha = 'right', fontsize = 11)
    ax.legend(fontsize = 12)
    ax.grid(axis = 'y', alpha = 0.3, linestyle = '--')
    
    #       Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}', ha = 'center', va = 'bottom', fontsize = 9, fontweight = 'bold')
    plt.tight_layout()
    plt.savefig(f'D:/VSCode/Scripts/Python/AQIPredictor/data/Trained Models/{city}/Graphs/{city.lower()}_model_comparison_bar.png', dpi = 300)
    plt.show()

def plotAllPredictions(metrics, city = 'Tokyo', num_samples = 100):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,11))
    
    #       Get Sequence length for fair comparisons
    min_length = min(len(metric['actuals']) for metric in metrics.values())
    num_samples = min(num_samples, min_length)
    
    #       Colors for each model
    colors = {
        'LSTM': '#2196F3', #Blue
        'Persistence': '#4CAF50', #Green
        'MA7': '#FF9800', #Orange
        'MA30': '#9C27B0', #Purple
        'SMM': '#F44336' #Red
    }
    
    #       Plot 1: Time Series comparison
    for idx, (model_name, metric) in enumerate(metrics.items()):
        actuals = metric['actuals'][:num_samples]
        predictions = metric['predictions'][:num_samples]
        
        if idx == 0:
            ax1.plot(actuals, label = 'Actual', color = 'black', linewidth = 2.5, alpha = 0.9, linestyle = '-', zorder = 100)
        ax1.plot(predictions, label = model_name, color = colors.get(model_name, '#777777'), linewidth = 2, alpha = 0.7, linestyle = '--')
    
    ax1.set_xlabel('Time Step (Hours)', fontsize = 12, fontweight = 'bold')
    ax1.set_ylabel('PM2.5 Value (µm/m³)', fontsize = 12, fontweight = 'bold')
    ax1.set_title(f'Predicted vs Actual PM2.5 Values in {city}', fontsize = 14, fontweight = 'bold')
    ax1.legend(fontsize = 11, loc = 'best', framealpha = 0.9)
    ax1.grid(True, alpha = 0.3, linestyle = '--')
    
    #       Plot 2: Scatter Plot comparison
    for model_name, metric in metrics.items():
        actuals = metric['actuals'][:num_samples]
        predictions = metric['predictions'][:num_samples]
        
        ax2.scatter(actuals, predictions, label = model_name, color = colors.get(model_name, '#777777'), alpha = 0.6, s = 25, edgecolors = 'white', linewidths = 0.5)
        
        #       Perfect Prediction line
        all_actuals = np.concatenate([metric['actuals'][:num_samples] for metric in metrics.values()])
        min_val, max_val = all_actuals.min(), all_actuals.max()
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth = 2.5, label = 'Perfect Prediction', zorder = 1000)
        ax2.set_xlabel('Actual PM2.5 Value (µm/m³)', fontsize = 12, fontweight = 'bold')
        ax2.set_ylabel('Predicted PM2.5 Value (µm/m³)', fontsize = 12, fontweight = 'bold')
        ax2.set_title(f"Prediction Accuracy - All Models ({city})", fontsize = 14, fontweight = 'bold')
        ax2.legend(fontsize = 11, loc = 'best', framealpha = 0.9)
        ax2.grid(True, alpha = 0.3, linestyle = '--')
        
        plt.tight_layout()
    plt.savefig(f'D:/VSCode/Scripts/Python/AQIPredictor/data/Trained Models/{city}/Graphs/{city.lower()}_all_models_predictions.png', dpi = 300)
    plt.show()
    
    print("All predictions comparison saved successfully")

def plotResiduals(metrics, city = 'Tokyo', num_samples = 100):
   """
   Plot Residual analysis for all models
   Residuals = Actual - Predicted
   
   Ars:
        metrics (dict): Dictionary containing model evaluation metrics
        city (str): City name for labeling
        num_samples (int): For equitable comparison, num samples defaults to 100
   """
   fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (16,12))
   
   min_length = min(len(metric['actuals']) for metric in metrics.values())
   num_samples = min(num_samples, min_length)
   
   colors = {
        'LSTM': '#2196F3', #Blue
        'Persistence': '#4CAF50', #Green
        'MA7': '#FF9800', #Orange
        'MA30': '#9C27B0', #Purple
        'SMM': '#F44336' #Red
   }
   
   #        Residual Plot 1: Time Series Residuals
   
   for model_name, metric in metrics.items():
       actuals = metric['actuals'][:num_samples]
       predictions = metric['predictions'][:num_samples]
       residuals = actuals - predictions
       
       ax1.plot(residuals, label = model_name, color = colors.get(model_name, '#777777'), linewidth = 1.5, alpha = 0.7)
   ax1.axhline(y=0, color = 'black', linestyle = '--', linewidth = 2, alpha = 0.5)
   ax1.set_xlabel('Time Step (Hours)', fontsize = 11, fontweight = 'bold')
   ax1.set_ylabel('Residual (µm/m³)', fontsize = 11, fontweight = 'bold')
   ax1.set_title(f'Residuals over Time', fontsize = 13, fontweight = 'bold')
   ax1.legend(fontsize = 9)
   ax1.grid(True, alpha = 0.3)
   
   #        Residual Plot 2: Residual Distribution (Histogram)
   
   for model_name, metric in metrics.items():
       actuals = metric['actuals'][:num_samples]
       predictions = metric['predictions'][:num_samples]
       residuals = actuals - predictions
       
       ax2.hist(residuals, bins = 30, alpha = 0.5, label = model_name, color = colors.get(model_name, '#777777'), edgecolor = 'white', linewidth = 0.5)
   ax2.axvline(x = 0, color = 'black', linestyle = '--', linewidth = 2, alpha = 0.7)
   ax2.set_xlabel('Residual (µm/m³)', fontsize = 11, fontweight = 'bold')
   ax2.set_ylabel('Frequency', fontsize = 11, fontweight = 'bold')
   ax2.set_title(f'Residual Distribution', fontsize = 13, fontweight = 'bold')
   ax2.legend(fontsize = 9)
   ax2.grid(True, alpha = 0.3, axis = 'y')
   
   #        Residual Plot 3: Residuals vs Actuals (heteroscedasticity check)
   for model_name, metric in metrics.items():
       actuals = metric['actuals'][:num_samples]
       predictions = metric['predictions'][:num_samples]
       residuals = actuals - predictions
       
       ax3.scatter(predictions, residuals, alpha = 0.5, s = 20, label = model_name ,color = colors.get(model_name, '#777777'), edgecolors = 'white', linewidths = 0.3)
   ax3.axhline(y = 0, color = 'black', linestyle = '--', linewidth = 2, alpha = 0.7)
   ax3.set_xlabel('Predicted PM2.5  (µm/m³)', fontsize = 11, fontweight = 'bold')
   ax3.set_ylabel('Residual (µm/m³)', fontsize = 11, fontweight = 'bold')
   ax3.set_title(f'Residuals vs Predicted', fontsize = 13, fontweight = 'bold')
   ax3.legend(fontsize = 9)
   ax3.grid(True, alpha = 0.3)
   
   #        Residual Plot 4: Box Plot of Residuals
   residual_data = []
   labels = []
   
   for model_name, metric in metrics.items():
       actuals = metric['actuals'][:num_samples]
       predictions = metric['predictions'][:num_samples]
       residuals = actuals - predictions
       residual_data.append(residuals)
       labels.append(model_name)
   bp = ax4.boxplot(residual_data, labels = labels, patch_artist = True, showmeans = True, meanline = True)
   
   #        Color the boxes
   for patch, model_name in zip(bp['boxes'], labels):
       patch.set_facecolor(colors.get(model_name, '#777777'))
       patch.set_alpha(0.6)

   ax4.axhline(y = 0, color = 'red', linestyle = '--', linewidth = 2, alpha = 0.7)
   ax4.set_xlabel('Model', fontsize = 11, fontweight = 'bold')
   ax4.set_ylabel('Residual (µm/m³)', fontsize = 11, fontweight = 'bold')
   ax4.set_title('Residual Distribution by Model', fontsize = 13, fontweight = 'bold')
   ax4.grid(True, alpha = 0.3, axis = 'y')
   plt.setp(ax4.xaxis.get_majorticklabels(), rotation = 20, ha = 'right')
   
   plt.suptitle(f'Residual Analysis - {city}', fontsize = 16, fontweight = 'bold', y = 0.995)
   plt.tight_layout()
   plt.savefig(f'D:/VSCode/Scripts/Python/AQIPredictor/data/Trained Models/{city}/Graphs/{city.lower()}_residual_analysis.png', dpi = 300)
   
   plt.show()
   
   print("Residual analysis plots saved successfully")

def plotImprovementPercentage(metrics, city = 'Tokyo', baseline = 'PERSISTENCE'):
    baseline_mae = metrics[baseline]['MAE']
    baseline_rmse = metrics[baseline]['RMSE']
    
    models = [m for m in metrics.keys() if m != baseline]
    mae_improvements = []
    rmse_improvements = []
    
    for model in models:
        mae_imp = ((baseline_mae - metrics[model]['MAE']) / baseline_mae) * 100
        rmse_imp = ((baseline_rmse - metrics[model]['RMSE']) / baseline_rmse) * 100
        mae_improvements.append(mae_imp)
        rmse_improvements.append(rmse_imp)
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize = (12, 7))
    
    bars1 = ax.bar(x - width/2, mae_improvements, width, label = 'MAE Improvement (%)', color = ['#4CAF50' if v > 0 else '#F44336' for v in rmse_improvements] , alpha = 0.8, edgecolor = 'black', linewidth = 1.5)
    bars2 = ax.bar(x + width/2, rmse_improvements, width, label = 'RMSE Improvement (%)', color = ['#2196F3' if v > 0 else '#FF9800' for v in rmse_improvements], alpha = 0.8, edgecolor = 'black', linewidth = 1.5)
    
    ax.axhline(y = 0, color = 'black', linestyle = '--', linewidth = 1)
    ax.set_xlabel('Model', fontsize = 14, fontweight = 'bold')
    ax.set_ylabel('Improvement (%)', fontsize = 13, fontweight = 'bold')
    ax.set_title(f'Model Improvement over {baseline} - {city}', fontsize = 15, fontweight = 'bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation = 15, ha = 'right', fontsize = 11)
    ax.legend(fontsize = 12)
    ax.grid(axis = 'y', alpha = 0.3, linestyle = '--')
    
    #       Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:+.1f}%', ha = 'center', va = 'bottom' if height > 0 else 'top', fontsize = 10, fontweight = 'bold')
    plt.tight_layout()
    plt.savefig(f'D:/VSCode/Scripts/Python/AQIPredictor/data/Trained Models/{city}/Graphs/{city.lower()}_improvement_percentage.png', dpi = 300)
    plt.show()
    
    print("Improvement chart saved successfully")

def printFinalResults(metrics , city = 'Tokyo'):
    print(f"\n\n{'='*80}")
    print(f"FINAL RESULTS SUMMARY - {city.upper()}")
    print(f"{'='*80}\n")
    
    #       Table header
    print(f"{'Model':<20} {'MAE':>12} {'RMSE': >12} {'R²':>10} {'Improvement':>15}")
    print(f"{'-'*80}")
    
    #       Get Persistence Baseline for Comparison
    persistence_mae = metrics['PERSISTENCE']['MAE']
    
    #       Print each model's results
    for model_name, metric in metrics.items():
        mae = metric['MAE']
        rmse = metric['RMSE']
        r2 = metric['R2'] if metric['R2'] is not None else float('nan')
        
        #       Calculate Improvement
        if model_name == 'PERSISTENCE':
            improvement = '-'
        else:
            imp_pct = ((persistence_mae - mae) / persistence_mae) * 100
            improvement = f"{imp_pct:.2f}%"
        
        #       Format R²
        r2_str = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
        
        print(f"{model_name:<20} {mae:>10.2f} {rmse:>10.2f} {r2_str:>10} {improvement:>15}")
    print(f"{'='*80}\n")
    
    #       Best Model Analysis
    best_mae_model = min(metrics.items(), key = lambda x: x[1]['MAE'])
    best_rmse_model = min(metrics.items(), key = lambda x: x[1]['RMSE'])
    
    print("BEST PERFORMERS:")
    print(f"    Lowest MAE: {best_mae_model[0]} with MAE = {best_mae_model[1]['MAE']:.2f} µm/m³")
    print(f"    Lowest RMSE: {best_rmse_model[0]} with RMSE = {best_rmse_model[1]['RMSE']:.2f} µm/m³")
    
    #       LSTM Specific
    lstm_mae = metrics['LSTM']['MAE']
    lstm_imp = ((persistence_mae - lstm_mae) / persistence_mae) * 100
    
    print("\nLSTM PERFORMANCE:")
    print(f"    MAE: {lstm_mae:.2f} µm/m³")
    print(f"    RMSE: {metrics['LSTM']['RMSE']:.2f} µm/m³")
    print(f"    R²: {metrics['LSTM']['R2']:.4f}")
    print(f"    Improvement over Persistence MAE: {lstm_imp:.2f}%")
    
    if lstm_imp > 0:
        print("    The LSTM model outperforms simple baselines.")
    else:
        print("    The LSTM model failed to outperform simple baselines.")
    
    print(f"\n{'='*80}\n")
    
    #       Residual Statistics
    print("RESIDUAL STATISTICS (LSTM):")
    lstm_residuals = metrics['LSTM']['actuals'] - metrics['LSTM']['predictions']
    print(f"    Mean Residual: {np.mean(lstm_residuals):.2f} µm/m³")
    print(f"    Std Dev of Residuals: {np.std(lstm_residuals):.2f} µm/m³")
    print(f"    Min Residual: {np.min(lstm_residuals):.2f} µm/m³")
    print(f"    Max Residual: {np.max(lstm_residuals):.2f} µm/m³")
    print(f"    Median Abs Residual: {np.median(np.abs(lstm_residuals)):.2f}µm/m³")
    
    print(f"\n{'='*80}\n")
if __name__ == "__main__":
    #Uncomment the city(ies) you want to train a model for
    #main(city = 'Tokyo')
    #main(city = 'Beijing')
    #main(city = 'Delhi')
    #main(city = 'Berlin')
    #main(city = 'London')
    #main(city = 'Paris')
    #main(city = 'Warsaw')