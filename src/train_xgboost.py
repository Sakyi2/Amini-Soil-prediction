import pandas as pd
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
import time
import optuna
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')



#import the train features 
X = pd.read_csv('X_processed.csv')

#import the train labels
y = pd.read_csv('y_processed.csv')

#import the test features
X_test=pd.read_csv("X_test_processed.csv")

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




#LETS EXPLORE THE DATA TO ADVANCE LEVEL
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
   
# Define objective function for XGBoost hyperparameter tuning

def objective_xgb(trial):
    # Define hyperparameters to tune
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42
    }
    
    # Create XGBoost MultiOutputRegressor
    xgb_model = MultiOutputRegressor(xgb.XGBRegressor(**params))
    
        # Train the model

    xgb_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = xgb_model.predict(X_val)
    
    # Calculate MAE
    mae = mean_absolute_error(y_val, y_pred)
    
    return mae

# Run the hyperparameter optimization
print("Tuning XGBoost hyperparameters...")
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=10)  # Adjust n_trials as needed

print("Best XGBoost Parameters:", study_xgb.best_params)
print("Best XGBoost MAE:", study_xgb.best_value)

# Create the optimized XGBoost model
best_xgb_model = MultiOutputRegressor(xgb.XGBRegressor(**study_xgb.best_params, random_state=42))

# Evaluate XGBoost model
xgb_results = evaluate_model(best_xgb_model, X_train, y_train, X_val, y_val, "XGBoost")