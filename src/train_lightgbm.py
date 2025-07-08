import lightgbm as lgb
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import KFold
import time
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor

import warnings
warnings.filterwarnings('ignore')

X = pd.read_csv('X_processed.csv')
X_test = pd.read_csv('X_test_processed.csv')
y = pd.read_csv('y_processed.csv')

# Drop if still in the data
if 'PID' in X.columns:
    X = X.drop(columns=['PID'])
if 'site' in X.columns:
    X = X.drop(columns=['site'])

if 'PID' in X_test.columns:
    X_test = X_test.drop(columns=['PID'])
if 'site' in X_test.columns:
    X_test = X_test.drop(columns=['site'])


#split the data into training and validation sets 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

def evaluate_model(model,X_train,y_train,X_val,y_val,model_name,):
      #tracking training time 
    StartTime = time.time()
    
    #fit the model
    model.fit(X_train,y_train)

    #trainin the time
    trainTime = time.time()-StartTime

    #prediction
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)


    #check the errors
    train_mae = mean_absolute_error(y_train,y_pred_train)
    val_mae = mean_absolute_error(y_val,y_pred_val)


    #check the RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train,y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val,y_pred_val))
   

     # Print results
    print(f"\n{model_name} Results:")
    print(f"Training Time: {trainTime:.2f} seconds")
    print(f"Training MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
    print(f"Validation MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")

    # Return the results
    return {
        'model': model,
        'name': model_name,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_time': trainTime
    }








# Define objective function for LightGBM hyperparameter tuning
def objective_lgb(trial):
    # Define hyperparameters to tune
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', -1, 15),  # -1 means no limit
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42
    }
    
    # Create LightGBM MultiOutputRegressor
    lgb_model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
    
    # Train the model
    lgb_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lgb_model.predict(X_val)
    
    # Calculate MAE
    mae = mean_absolute_error(y_val, y_pred)
    
    return mae

# Run the hyperparameter optimization
print("Tuning LightGBM hyperparameters...")
study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=10)  # Adjust n_trials as needed

print("Best LightGBM Parameters:", study_lgb.best_params)
print("Best LightGBM MAE:", study_lgb.best_value)

# Create the optimized LightGBM model
best_lgb_model = MultiOutputRegressor(lgb.LGBMRegressor(**study_lgb.best_params, random_state=42))

# Evaluate LightGBM model
lgb_results = evaluate_model(best_lgb_model, X_train, X_val, y_train, y_val, "LightGBM")