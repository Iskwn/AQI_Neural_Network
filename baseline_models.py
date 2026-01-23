import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def persistence_model(train_df, test_df):
    """
    Baseline Model #1: Persistence Model
    Predicts that Tommorow's AQI = Today's AQI

    Args:
        train_df: Not used but kept for consistency
        test_df: Test data with 'value' column
    Returns:
        predictions, actuals, metrics
    """
    #Prediction Tomorrow's value = today's value
    actuals = test_df.values[1:] # Tomorrow's actual values
    predictions = test_df.values[:-1] # Today's values (shifted)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    try:
        r2 = r2_score(actuals, predictions)
    except:
        r2 = None
        
    metrics = {
        'name': 'Persistence',
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'predictions': predictions,
        'actuals': actuals
    }
    
    return predictions, actuals, metrics


def moving_average_model(train_df, test_df, window = 7):
    """
    Baseline Model #2: Moving Average
    Predicts tomorrow's value = average of N days
    Args:
        train_df: Training data
        test_df: Test data
        window: Number of days to average out (default of 7 for a week)
    Returns:
        predictions, actuals, metrics
    """
    #Combine train and test data sets for rolling calculations
    all_data = pd.concat([train_df, test_df], ignore_index = True)
    
    #Calculate moving average
    print(type(all_data))
    all_data['ma'] = all_data.rolling(window=window).mean()
    
    #Get predictions for testing period
    test_start_idx = len(train_df)
    predictions = all_data['ma'].iloc[test_start_idx:].values
    actuals = test_df.values
    
    #Remove non-values (from initial 'Window' predictions)
    valid_mask = ~np.isnan(predictions)
    predictions = predictions[valid_mask]
    actuals = actuals[valid_mask]
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    try:
        r2 = r2_score(actuals, predictions)
    except:
        r2 = None
    
    metrics = {
        'name': f'Moving Average ({window}-day)',
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'predictions': predictions,
        'actuals': actuals
    }
    
    return predictions, actuals, metrics

def simple_mean_model(train_df, test_df):
    """
    Baseline Model #3: Mean Model
    Predicted value = mean of training data
    Args:
        train_df: Trianing data
        test_df: Test data
    Returns:
    predictions, actuals, metrics
    """
    
    train_mean = train_df.mean()
    actuals = test_df.values
    predictions = np.full_like(actuals, train_mean)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    try:
        r2 = r2_score(actuals, predictions)
    except:
        r2 = None
        
    metrics = {
        'name': 'Mean Prediction',
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'predictions': predictions,
        'actuals': actuals
    }
    
    return predictions, actuals, metrics