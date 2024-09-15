'''
FILE:       Individual_FDS.py
PROJECT:    The Applicability of Feature Distribution Smoothing in Imbalanced Regression
AUTHOR:     Eric Lin
'''

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import RBF
from scipy.ndimage import gaussian_filter1d
import time


#Constants
FILE_PATH = 'canola_data.csv'
YIELD_BIN_LABELS = ['low', 'medium', 'high', 'very high']
TYPE_BIN_LABELS = ['basal', 'HER', 'cell_line', 'normal', 'luminal_A', 'luminal_B' ]
YIELD_LABEL = 'yield_label'
TYPE_LABEL = 'type'
# Set the standard deviation for the Gaussian kernel
SIGMA = 1.0
#Set the number of features
FEATURE_NUMBER = 200

#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
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

# Fixedly pick features
def cut_database(df):
    number_of_features = len(df.columns) - FEATURE_NUMBER
    df = df.iloc[:, number_of_features:]
    return df

# Randomly pick features
def random_select_features(df):
    # print('\nFeature Amount: ', FEATURE_NUMBER)
    columns_without_last = df.columns[:-1]
    selected_columns = np.random.choice(columns_without_last, size=FEATURE_NUMBER-1, replace=False)
    selected_columns = np.append(selected_columns, df.columns[-1])
    selected_features_df = df[selected_columns]
    return selected_features_df

#Bin the dataset
#Based on the yield of the canola into four categories: Low, Medium, High, and Very High.
# Low: Yields in the lowest 25% of the dataset.
# Medium: Yields between the 25th and 50th percentile.
# High: Yields between the 50th and 75th percentile.
# Very High: Yields in the highest 25% of the dataset.
def yield_bin(df):
    #sort the dataset based on yield
    sorted_data = df.sort_values(by='yld')
    #calculate the percentiles of the yield data
    yield_percentiles = sorted_data['yld'].quantile([0.25, 0.5, 0.75])
    #bin the sorted data based on the calculated percentiles
    sorted_data['yield_label'] = pd.cut(sorted_data['yld'], 
                                        bins = [-np.inf, yield_percentiles[0.25], yield_percentiles[0.5], yield_percentiles[0.75], np.inf], 
                                        labels=YIELD_BIN_LABELS)
    # Display the sorted DataFrame with the new 'yield_label' column
    sorted_data.reset_index(drop=True, inplace=True)  # Optional: reset index after sorting
    return sorted_data
    

# Calculatge the Mean Vector
#This will give a mean vector μb of size 1*number of features, 
#where each component is the average of a particular feature across all entites.
def calculate_mean(df, dataset):
    mean_vectors = []
    if dataset == 1: 
        bin_label_name = YIELD_LABEL
        bin_sets = YIELD_BIN_LABELS
    else: 
        bin_label_name = TYPE_LABEL
        bin_sets = TYPE_BIN_LABELS
        
    for bin_label in bin_sets:
        #bin_data = df[df['yield_label'] == bin_label].iloc[:, :-1]  # Exclude the bin column
        bin_data = df[df[bin_label_name] == bin_label].iloc[:, :-1]
        mu_b = bin_data.mean()  #calculate the mean
        mean_vectors.append(mu_b)
    mean_vectors_df = pd.DataFrame(mean_vectors, index=bin_sets)
    return mean_vectors_df


# Calculate the Covariance Matrix
#The resulting matrix will have a size of m×m if there are m variables, 
#with the diagonal elements representing the variances of the individual variables, 
#and the off-diagonal elements representing the covariances between pairs of variables.
def calculate_covariance(df, mean_vector, dataset):
    covariance_matrix = {}
    
    if dataset == 1: 
        bin_label_name = YIELD_LABEL
        bin_sets = YIELD_BIN_LABELS
    else: 
        bin_label_name = TYPE_LABEL
        bin_sets = TYPE_BIN_LABELS
    
    for bin_label in bin_sets:
        bin_data = df[df[bin_label_name] == bin_label].iloc[:, :-2]  # Exclude the bin column
        N_b = len(bin_data)
        if N_b > 1:  #To compute covariance, we need at least 2 entities
            #Sigma_b = bin_data.cov() * ((N_b - 1) / N_b)
            # Get the mean vector μb
            current_mean_vector = mean_vector.loc[bin_label]
            # *** Delete yield column to calculate covariance matrix
            current_mean_vector = current_mean_vector.iloc[:-1]
            #print(current_mean_vector)
            # Subtract the mean vector from each row to get the deviation scores
            deviation_scores = bin_data - current_mean_vector
            # Calculate the covariance matrix
            Sigma_b = deviation_scores.T.dot(deviation_scores) / (N_b - 1) #The deviation of each sample from the mean times the transpose of the deviation vector
            covariance_matrix[bin_label] = Sigma_b
        else:
            # If there's not enough data for covariance, use a DataFrame of NaNs
            covariance_matrix[bin_label] = pd.DataFrame(np.nan, index=bin_data.columns, columns=bin_data.columns)

    # Convert the dictionary of DataFrames into a MultiIndex DataFrame
    covariance_matrix_df = pd.concat(covariance_matrix, names=[YIELD_LABEL, 'feature'])
    return covariance_matrix_df

#================================================================================
def distribution_smooth(mean_vector, covariance_matrix, dataset):
    
    if dataset == 1:
        #Exclude the target values - yield column: 
        # save the yield column
        yield_col = mean_vector.iloc[:, -1]  # This selects the last column
        # save the column headers and bin headers 
        columns = mean_vector.columns[:-1]
        bins = mean_vector.index
        # Exclude the last column from DataFrame
        mean_vector = mean_vector.iloc[:, :-1]
    else:
        columns = mean_vector.columns
        bins = mean_vector.index

    # Apply the Gaussian filter from scipy
    smoothed_mean_vector = gaussian_filter1d(mean_vector, SIGMA)
    # Convert the smoothed mean vectors back to a dataframe
    smoothed_mean_vector_df = pd.DataFrame(smoothed_mean_vector, columns=columns, index=bins)
    
    if dataset == 1:
        # Add the yield column back
        smoothed_mean_vector_df['yld'] = yield_col
    
    # ---------------------- Matrix ------------------------------
    # save the column headers and bin headers 
    co_columns = covariance_matrix.columns
    co_bins = covariance_matrix.index
    # Apply the Gaussian filter from scipy
    smoothed_covariance_matrix = gaussian_filter1d(covariance_matrix, SIGMA)
    # Convert the smoothed mean vectors back to a dataframe
    smoothed_covariance_matrix_df = pd.DataFrame(smoothed_covariance_matrix, columns=co_columns, index=co_bins)
    # print(smooth_covariance_matrix_df.loc['very high'])
    
    return smoothed_mean_vector_df, smoothed_covariance_matrix_df


# Follow the standard whitening and re-coloring procedure to calibrate the feature representation for each input sample
def whitening_recoloring(raw_data, ori_mean_vector, ori_cov_matrix, smoothed_mean_vector, smoothed_cov_matrix, dataset):
    if dataset == 1: 
        bin_label_name = YIELD_LABEL
        bin_sets = YIELD_BIN_LABELS
    else: 
        bin_label_name = TYPE_LABEL
        bin_sets = TYPE_BIN_LABELS
    
    if dataset == 1:
        whitened_recolored_data_matrix_buffer = yield_bin(raw_data)
    else:
        whitened_recolored_data_matrix_buffer = raw_data
    whitened_recolored_data_array_list = []
    # Smooth each feature in the feature space based on the mean vector and covariance matrix of the bin it located in
    for bin_label in bin_sets:
        print("[FDS] Whitening and Recoloring " + bin_label + " bin...")
        start_time = time.time()
        # Exclude the bin column
        bin_ori_cov_matrix = ori_cov_matrix.loc[bin_label]
        bin_smoothed_cov_matrix = smoothed_cov_matrix.loc[bin_label]
        # Convert the dataframes into numpy arrays
        bin_ori_cov_array = bin_ori_cov_matrix.to_numpy()
        bin_smoothed_cov_array = bin_smoothed_cov_matrix.to_numpy()
        
        # --------------- Inverse square root of Origin Covariance matrix  ----------------
        # Compute eigenvalue decomposition for Origin Matrix
        eigvals_ori, eigvecs_ori = np.linalg.eigh(bin_ori_cov_array)
        # *** Adjust negative eigenvalues to small positive values ***
        epsilon = 1e-10  # Very Small positive constant
        eigvals_ori_adjusted = np.where(eigvals_ori < 0, epsilon, eigvals_ori)
        # Compute inverse square root
        inv_sqrt_eigvals_ori = 1 / np.sqrt(eigvals_ori_adjusted)
        inv_sqrt_ori = eigvecs_ori @ np.diag(inv_sqrt_eigvals_ori) @ eigvecs_ori.T

        # --------------- Square root of Smoothed Covariance matrix  ----------------
        # Compute eigenvalue decomposition for Smoothed Matrix
        eigvals_smoothed, eigvecs_smoothed = np.linalg.eigh(bin_smoothed_cov_array)
        # *** Adjust negative eigenvalues to small positive values ***
        eigvals_smoothed_adjusted = np.where(eigvals_smoothed < 0, epsilon, eigvals_smoothed)
        # Compute square root
        sqrt_eigvals_smoothed = np.sqrt(eigvals_smoothed_adjusted)
        sqrt_smoothed = eigvecs_smoothed @ np.diag(sqrt_eigvals_smoothed) @ eigvecs_smoothed.T

        # --------------- Multiply Smoothed^(1/2) and Origin^(-1/2) ----------------
        whitened_matrix = inv_sqrt_ori @ sqrt_smoothed
        #print(whitened_matrix)
        
        # --------------- (z − µb) + µ˜b ----------------
        # convert necessary data into numpy arrays
        current_bin_ori_mean = ori_mean_vector.loc[bin_label].to_numpy()[:-1]
        current_bin_smoothed_mean = smoothed_mean_vector.loc[bin_label].to_numpy()[:-1]
        if dataset == 1:
            current_raw_features = yield_bin(raw_data).to_numpy()
        else:
            current_raw_features = raw_data.to_numpy()
        # Iterate over each feature in the feature space of the entire dataset
        for index, features in enumerate(current_raw_features):    #except the last column - yld
            if features[-1] == bin_label: # IF it is the same feature
                z = features[:-2]   # extract all feature values of a data sample except yield and bin columns
                z_minus_bin_mean = z - current_bin_ori_mean # Calculate the standard deviation
                z_whitened_recolored = (whitened_matrix @ z_minus_bin_mean) + current_bin_smoothed_mean # (z − µb) + µ˜b
                whitened_recolored_data_array_list.append(z_whitened_recolored)
        
        end_time = time.time()
        print(f"      Time used: {round(end_time - start_time, 3)} seconds.")
    whitened_recolored_data = np.array(whitened_recolored_data_array_list)
    
    # ***Scale down the entire feature space based on the size of the max feature value***
    max_value = whitened_recolored_data.max()
    # *The size of processed feature values becomes larger as the feature quantity becomes larger
    whitened_recolored_data = whitened_recolored_data / whitened_recolored_scale_down(max_value)
    
    # Place the whitened and recolored feature data back into the data frame
    whitened_recolored_data_matrix_buffer.iloc[:, :-2] = whitened_recolored_data
    # print("\n[FDS] Whitened and Recolored data samples: ")
    # print(whitened_recolored_data_matrix_buffer)
    return whitened_recolored_data_matrix_buffer

# A function to scale down the size of the values of whitened and recolored feature values
def whitened_recolored_scale_down(data):
    scale = 10 ** (int(np.log10(data)))
    return scale

# Import and extract raw data based on the specific feature number
def get_raw_data(file):
    df = get_data(file)
    sorted_df = random_select_features(df)
    return sorted_df

# Get, sort and shorten database
def original_mean_covariance(sorted_df, dataset, output):
    # #Get data from database
    # df = get_data(file)
    # #print('[SUCCESS] '+file+" imported.")
    
    # #Shorten database
    # sorted_df = random_select_features(df)
    # #print('[SUCCESS] Dataset shortened.')   
    
    #Sort the dataframe
    #print('[SUCCESS] Dataset sorted.') 
    
    #Calculate mean vectors and covariance matrices
    mean_vector = calculate_mean(sorted_df, dataset)
    #print('[SUCCESS] Mean vecotors calculated.')
    covariance_matrix = calculate_covariance(sorted_df, mean_vector, dataset)
    #print('[SUCCESS] Covariance Matrices calculated.')

    if(output == 1):
        print("Mean Vectors")
        print(mean_vector)
        print("--------------------------------------------------------------------------------------------------------")
        print("Covariance Matrices")
        print(covariance_matrix)

    return mean_vector, covariance_matrix

# FDS
def fds(raw_data, dataset):
    # try:
        sorted_df = raw_data
        if dataset == 1:
            sorted_df = yield_bin(raw_data)
            
        print("[FDS] Processing Raw Mean Vectors and Covariance Matrices Processing...")
        mean_vector, covariance_matrix = original_mean_covariance(sorted_df, dataset, 0)
        
        print("[FDS] Smoothing Mean Vectors and Covariance Matrices...")
        smoothed_mean_vectors, smoothed_covariance_matrix = distribution_smooth(mean_vector, covariance_matrix, dataset)
        whitened_recolored_df = whitening_recoloring(raw_data, mean_vector, covariance_matrix, smoothed_mean_vectors, smoothed_covariance_matrix, dataset)
        return whitened_recolored_df
    # except Exception as e:
    #     print(f"[FDS ERROR] {e}")
        
#================================================================================
#Main Function
def main():
    #fds(FILE_PATH)
    raw_data = get_data(FILE_PATH)
    raw_data = random_select_features(raw_data)
    mean_vector, covariance_matrix = original_mean_covariance(FILE_PATH, 0)
    smoothed_mean_vector, smoothed_covariance_matrix = distribution_smooth(mean_vector, covariance_matrix)
    whitening_recoloring(raw_data, mean_vector, covariance_matrix, smoothed_mean_vector, smoothed_covariance_matrix)

#Run
if __name__ == '__main__':
    main()