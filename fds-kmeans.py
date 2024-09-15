from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import time
from Individual_FDS import fds
from k_mean import k_means
from visualization import kmeans_plot

FILE_PATH = 'Breast_GSE45827.csv'
FEATURE_NUMBER = 200

def use_column(col):
    columns_to_exclude = ['samples']
    return col not in columns_to_exclude

def fds_kmeans(feature_number):
    set_feature_number(feature_number)
    print('Feature Amount: ', feature_number)
    
    raw_data = get_raw_data(FILE_PATH)
    print("RAW: ")
    raw_k, raw_pca, raw_ori_pca= k_means(raw_data)
    smoothed_df = fds(raw_data, 2)
    print("FDS: ")
    smoothed_k, smoothed_pca, smoothed_ori_pca = k_means(smoothed_df)
    print("\n------------------- FDS Evaluations --------------------\n")
    print(f"[K-means Inertia] Raw: {raw_k.inertia_}; FDS: {smoothed_k.inertia_}; Inertia reduced: {round((raw_k.inertia_ - smoothed_k.inertia_ ) / raw_k.inertia_ *100, 2)}%")
    kmeans_plot(raw_data, smoothed_df, raw_k, smoothed_k, raw_pca, smoothed_pca, raw_ori_pca, smoothed_ori_pca)

def set_feature_number(feature_number): 
    global FEATURE_NUMBER
    FEATURE_NUMBER = feature_number

#Read the data from the external dataset
def get_data(file_path):
    df = pd.read_csv(file_path, usecols = use_column)
    return df

# Import and extract raw data based on the specific feature number
def get_raw_data(file):
    df = get_data(file)
    # Remove the 'type' column and store it
    type_column = df.pop('type')
    # Assign it back to the DataFrame; this will add it as the last column
    df['type'] = type_column
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

#================================================================================
#Main Function
def main():
    feature_number = FEATURE_NUMBER
    if len(sys.argv) == 2:
        feature_number = int(sys.argv[1])
        fds_kmeans(feature_number)

    else:
        print("[ERROR] Wrong arguments.")



#Run
if __name__ == '__main__':
    main()