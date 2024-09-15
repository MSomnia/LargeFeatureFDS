'''
FILE:       svrm.py
PROJECT:    The Applicability of Feature Distribution Smoothing in Imbalanced Regression
AUTHOR:     Eric Lin
'''
# Import necessary libraries
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ndcg_score

#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------

def svrm(df):
    # Remove yield label column in the dataframe
    if df.columns[-1] == 'yield_label':
        df = df.drop('yield_label', axis=1)
    # Split the dataset into features (X) and the target variable (y)
    X = df.iloc[:, :-1]  # Assumes the last column is the target
    y = df.iloc[:, -1]
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    # Current bins: 4; Train set size = 3, Test size = 1

    # Feature Scaling
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train_scaled = sc_X.fit_transform(X_train)
    y_train_scaled = sc_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    # Train the SVR model
    svr = SVR(kernel='rbf')  # Using Radial Basis Function kernel
    svr.fit(X_train_scaled, y_train_scaled)
    
    # Predicting on the test set
    X_test_scaled = sc_X.transform(X_test)
    y_pred_scaled = svr.predict(X_test_scaled)
    # Transforming the scaled prediction back to original scale
    y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # -------------------------- Evaluate the model --------------------------
    # MSE - Mean Squared Error
    # Measure of the quality of a regression model, indicating the average squared difference between the actual and predicted values.
    # print("[SVR] Test sets:\n", y_test)
    # print("\n[SVR] Predictions on the test sets:\n", y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Pearson Correlation
    corr, _ = pearsonr(y_test, y_pred)
    
    # NDCG
    y_reshaped = y.to_numpy().reshape(-1, 1)
    NDCG_y = MinMaxScaler().fit_transform(y_reshaped).ravel()  # Scale target to [0, 1] for relevance
    X_train, X_test, y_train, y_test = train_test_split(X, NDCG_y, test_size=0.2, random_state=42)
    model = SVR()
    model.fit(X_train, y_train)
    NDCG_y_pred = model.predict(X_test)
    
    # Reshape the predictions and actual values for ndcg_score input requirements
    y_pred_rescaled = MinMaxScaler().fit_transform(NDCG_y_pred.reshape(-1, 1)).ravel()  # Ensure predictions are scaled to [0, 1]
    y_test_rescaled = MinMaxScaler().fit_transform(y_test.reshape(-1, 1)).ravel()  # Ensure actuals are scaled to [0, 1]

    # Calculate NDCG score
    ndcg = ndcg_score([y_test_rescaled], [y_pred_rescaled])
    
    return mse, corr, ndcg
    