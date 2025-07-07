#PREDICTING NUTRIENT GAPS

#this notebbok will guide u the process of building a machine learning a model to prediction nutrient gaps in soil

# Models Used in This Project

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')

#STEP 2 
#EXPLORE THE DATA ANAYSIS(EDA)

train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')
train_gap_df = pd.read_csv('Gap_Train.csv')
test_gap_df =pd.read_csv('Gap_Test.csv')
sample_submission_df =pd.read_csv('SampleSubmission.csv')

train_df.head()
train_gap_df.head()
test_df.head()

test_gap_df = pd.merge(test_gap_df, test_df[['PID', 'BulkDensity']], on='PID', how='left')
test_gap_df.head()

sample_submission_df.head()

#FEATURE SELECTION FOR AMINI SOIL PREDICTION
print("Train Data Info:")
print(train_df.info())
print("\nTest Data Info:")
print(test_df.info())

#visualize the K 
sns.histplot(train_df['K'] , bins=30,kde=True)
plt.title('Distribution of K')
plt.show()

#STEP 4 FEACTURE SELECTION AND PROCESSING
# check for missing values 
#FILLING MISSIG VALUES WITH COLUMN WITH MEAN
for column in train_df.columns:
    if train_df[column].isnull().any():
     train_df[column].fillna(train_df[column].mean(),inplace=True)

#fill missing values with columns mean
for column in test_df.columns:
     if test_df[column].isnull().any():
        test_df[column].fillna(test_df[column].mean(), inplace=True)

#SOME COLUMNS ARE INDENTIFIERS OR NOT USEFUL FOR PREDICTION
#DROP NON NUMERIC OR IRRELEVANT COLUMNS 
columns_to_drop = ['N', 'P', 'K', 'Ca', 'Mg', 'S', 'Fe', 'Mn', 'Zn', 'Cu', 'B']

X = train_df.drop(columns=columns_to_drop)
y = train_df[columns_to_drop]

#feature selection
X_test = test_df.drop(columns=['PID',"site"])


#SPLIT THE DATA INTO TRAIN AND TEST SETS 
X_train,X_val,y_train,y_val = train_test_split(X,y ,random_state=42)

#ENCODE TARGET VARIABLES 
from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
train_df.columns = Le.fit_transform(train_df.columns)

X_train =X_train.drop(columns=['PID','site'])
X_val = X_val.drop(columns=["PID",'site'])

#THIS PREDICTION HAS MULTILE TARGET COLMNS NOT JUST ON CLASS
#BUILD  THE MODEL
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

#prediction
pred = model.predict(X_test)
y_pred = model.predict(X_val)

# EVALUATION : CHECKING ACCURACY,CLASSIFICATION REPORT
#calulate the absolute error between the predicted and the actaul values
mae = mean_absolute_error(y_val,y_pred)
#calculate the average suare differnce between the predicted and actual values
mse  = mean_squared_error(y_pred,y_val)

#calculate the root mean square error
rmse = np.sqrt(mse)
print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}')

test_pred =model.predict(X_test)
#split the predictions
N_pred =  test_pred[:, 0]  # Predictions for N
P_pred =  test_pred[:, 1]  # Predictions for P
K_pred =  test_pred[:, 2]  # Predictions for K
Ca_pred = test_pred[:, 3]  # Predictions for Ca
Mg_pred = test_pred[:, 4]  # Predictions for Mg
S_pred =  test_pred[:, 5]  # Predictions for S
Fe_pred = test_pred[:, 6]  # Predictions for Fe
Mn_pred = test_pred[:, 7]  # Predictions for Mn
Zn_pred = test_pred[:, 8]  # Predictions for Zn
Cu_pred = test_pred[:, 9]  # Predictions for Cu
B_pred =  test_pred[:, 10] # predictions for B

#VISUALIZE THE PREDICTIONS IN EACH NUTIRENT
import matplotlib.pyplot as plt

nutrient_names = ['N', 'P', 'K', 'Ca', 'Mg', 'S', 'Fe', 'Mn', 'Zn', 'Cu', 'B']

plt.figure(figsize=(18, 10))
for i in range(11):
    plt.subplot(3, 4, i+1)
    plt.hist(test_pred[:, i], bins=30, color='skyblue', edgecolor='black')
    plt.title(f'{nutrient_names[i]} Predictions')
    plt.tight_layout()


#SETUP THE SUBMISSION FILE
#creating ur submiision file
submission = pd.DataFrame({
    'PID':test_df['PID'],
    'N': N_pred,
    'P': P_pred,
    'K': K_pred,
    'Ca': Ca_pred,
    'Mg': Mg_pred,
    'S': S_pred,
    'Fe': Fe_pred,
    'Mn': Mn_pred,
    'Zn': Zn_pred,
    'Cu': Cu_pred,
    'B': B_pred
})

submission.head()

#turn submission into a 3 column that PID,nutrient and values
submission_melted = submission.melt(id_vars=['PID'] , var_name='Nutrient',value_name='Available_Nutrients_ppm')
submission_melted = submission_melted.sort_values('PID')
submission_melted.head()



#STEP 5: PREDICTIING THE NUTRIENTG GAPS
#PREDICTING THE NUTRIENT GAPS

#merge test_gap.df with subimission_melted on pid and nutrient
nutrient_df = pd.merge(submission_melted,test_gap_df , on=['PID', 'Nutrient'],how='left')

soil_depth = 20 #cm
#caculate the avialable nutrients _in_kg_ha
nutrient_df['Available_Nutrients_in_kg_ha'] = (nutrient_df['Available_Nutrients_ppm'] * 
                                                soil_depth * nutrient_df['BulkDensity'] * 0.1)

nutrient_df.head()

#finding the gap
nutrient_df['Gap'] = nutrient_df['Required'] - nutrient_df['Available_Nutrients_in_kg_ha']
nutrient_df['ID'] = nutrient_df['Nutrient'] + '_' + nutrient_df['PID']
nutrient_df = nutrient_df[['ID','Gap']]
nutrient_df.head()