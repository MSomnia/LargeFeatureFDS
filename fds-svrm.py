'''
FILE:       fds-svrm.py
PROJECT:    The Applicability of Feature Distribution Smoothing in Imbalanced Regression
AUTHOR:     Eric Lin
'''
from Individual_FDS import fds
from svrm import svrm
from visualization import scatter
import sys
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Constants
FILE_PATH = 'canola_data.csv'
FEATURE_INCREMENT = 100
MIN_FEATURE = 100
MAX_FEATURE = 500
FEATURE_NUMBER = 200

global start_time
end_time = time.time()

#================================================================================
#Main Function
def main():
    feature_number = FEATURE_NUMBER
    if len(sys.argv) == 2:
        feature_number = int(sys.argv[1])
        results, raw_df, smoothed_df = fds_svrm(feature_number)
        scatter(raw_df, smoothed_df, results)
        
    elif len(sys.argv) == 4:
        feature_increment = FEATURE_INCREMENT
        min_feature = MIN_FEATURE
        max_feature = MAX_FEATURE
        feature_increment = int(sys.argv[3])
        min_feature = int(sys.argv[1])
        max_feature = int(sys.argv[2])
        performance(feature_increment, min_feature, max_feature)
    
    else:
        print("[ERROR] Wrong arguments.")
        
def fds_svrm(feature_number):
    set_feature_number(feature_number)
    print('Feature Amount: ', feature_number)
    
    raw_data = get_raw_data(FILE_PATH)
    
    smoothed_df = fds(raw_data, 1)
    
    # time.sleep(2)
    print("\n")
    # print('[SVR] Feature smoothed through FDS: ')
    start_time = time.time()
    print("[SVRM] Feature distribution smoothed dataset training...")
    smoothed_mse, smoothed_pear_corr, smoothed_ndcg= svrm(smoothed_df)
    time_counter(start_time)
    # print('\n\n****************************************')
    # print('[SVR] Raw feature: ')
    raw_df = raw_data
    start_time = time.time()
    print("[SVRM] Raw dataset training...")
    ori_mse, ori_pear_corr, ori_ndcg = svrm(raw_df)
    time_counter(start_time)
    
    # time.sleep(2)
    
    print("\n------------------- FDS Evaluations --------------------\n")
    # MSE
    reduced_mse = (ori_mse - smoothed_mse) / ori_mse * 100
    print(f"[Mean Squared Error] Raw: {ori_mse}; FDS: {smoothed_mse}; MSE reduced: {round(reduced_mse, 2)}%")
    # Pearson Correlation Coefficient
    reduced_pear = (smoothed_pear_corr - ori_pear_corr) / smoothed_pear_corr * 100
    print(f"[Pearson Correlation Coefficient] Raw: {ori_pear_corr}; FDS: {smoothed_pear_corr}; PCC increased: {round(reduced_pear, 2)}%")
    # NDCG
    reduced_ndcg = (smoothed_ndcg - ori_ndcg) / smoothed_ndcg * 100
    print(f"[NDCG] Raw: {ori_ndcg}; FDS: {smoothed_ndcg}; NDCG increased: {round(reduced_ndcg, 2)}%")
    result = {}
    result[0] = ori_mse
    result[1] = smoothed_mse
    result[2] = reduced_mse
    result[3] = ori_pear_corr
    result[4] = smoothed_pear_corr
    result[5] = reduced_pear
    result[6] = ori_ndcg
    result[7] = smoothed_ndcg
    result[8] = reduced_ndcg
    
    return result, raw_df, smoothed_df

def performance(feature_increment, min_feature, max_feature):
    mse_performance = []
    pear_performance = []
    ndcg_performance = []
    time_performance = []
    
    # Create subplots - one for each array
    fig, subplot = plt.subplots(2,2, figsize=(10, 15))

    for feature_number in range(min_feature, max_feature+1, feature_increment):
        print("\n\n===================================================================\n")
        start_time = time.time()
        results, raw_df, smoothed_df = fds_svrm(feature_number)
        time_used = time_counter(start_time)
        mse_performance.append(results[2])
        pear_performance.append(results[5])
        ndcg_performance.append(results[8])
        time_performance.append(time_used)
    
    # x_values = [FEATURE_INCREMENT * (x+1) for x in range(len(mse_performance))]
    # plt.plot(x_values, mse_performance, label='MSE')
    # plt.plot(x_values, pear_performance, label='PCC')
    # plt.plot(x_values, ndcg_performance, label='NDCG')
    # plt.title('Improvement of Machine Learning Performance by FDS Algorithm in difference features.')
    # plt.xlabel('Number of Features')
    # plt.ylabel('% Performance')
    # plt.show()
        
    # MSE line chart
    x_values = [feature_increment * x + min_feature for x in range(len(mse_performance))]
    subplot[0,0].plot(x_values, mse_performance)
    subplot[0,0].set_title(f'MSE Performance')
    subplot[0,0].set_xlabel('Number of Features')
    subplot[0,0].set_ylabel('% Performance')
    
    # PCC line chart
    subplot[0,1].plot(x_values, pear_performance)
    subplot[0,1].set_title(f'Pearson Correlation Coefficient Performance')
    subplot[0,1].set_xlabel('Number of Features')
    subplot[0,1].set_ylabel('% Performance')
    
    # NDCG line chart
    subplot[1,0].plot(x_values, ndcg_performance)
    subplot[1,0].set_title(f'NDCG Performance')
    subplot[1,0].set_xlabel('Number of Features')
    subplot[1,0].set_ylabel('% Performance')
    
    #Time
    subplot[1,1].plot(x_values, time_performance)
    subplot[1,1].set_title(f'Time Performance')
    subplot[1,1].set_xlabel('Number of Features')
    subplot[1,1].set_ylabel('Time (Seconds)')
    
    # Adjust layout so that titles and labels do not overlap
    plt.tight_layout()
    plt.show()
    print("[Line Chart] generated")


def set_feature_number(feature_number): 
    global FEATURE_NUMBER
    FEATURE_NUMBER = feature_number

#Exclude columns that won't be used in the dataset
def use_column(col):
    columns_to_exclude = ['No.', 'Name', 'flr', 'mat', 'ht', 'pc', 'oc', 'gluc', 'samples']
    return col not in columns_to_exclude

#Read the data from the external dataset
def get_data(file_path):
    df = pd.read_csv(file_path, usecols = use_column)
    return df

# Import and extract raw data based on the specific feature number
def get_raw_data(file):
    df = get_data(file)
    sorted_df = random_select_features(df)
    return sorted_df

# Randomly pick features
def random_select_features(df):
    # print('\nFeature Amount: ', FEATURE_NUMBER)
    columns_without_last = df.columns[:-1]
    selected_columns = np.random.choice(columns_without_last, size=FEATURE_NUMBER-1, replace=False)
    selected_columns = np.append(selected_columns, df.columns[-1])
    selected_features_df = df[selected_columns]
    return selected_features_df

def time_counter(start_time):
    end_time = time.time()
    time_used = round(end_time - start_time, 3)
    print(f"      Time used: {time_used} seconds.")
    return time_used

#Run
if __name__ == '__main__':
    main()