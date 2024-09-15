import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd

# A linear combination of all original features weighted by their contribution to this direction of maximum variance.
def scatter(df_raw, df_smoothed, results): 
    # Remove yield and label column
    df_raw = df_raw.iloc[:, :-1]
    df_smoothed = df_smoothed.iloc[:, :-2]
    
    # Step 1: Apply PCA to reduce dimensions to 2 for both datasets
    pca = PCA(n_components=2)
    raw_pca = pca.fit_transform(df_raw)
    processed_pca = pca.fit_transform(df_smoothed)
    explained_variance_ratio = pca.explained_variance_ratio_

    # Step 2: Create a scatter plot
    plt.figure(figsize=(10, 6))

    # Plot raw dataset
    plt.scatter(raw_pca[:, 0], raw_pca[:, 1], alpha=0.5, label='Raw Data', c='red')

    # Plot processed dataset
    plt.scatter(processed_pca[:, 0], processed_pca[:, 1], alpha=0.5, label='Processed Data', c='blue')

    # Enhancing the plot
    plt.title(f'PCA Comparison of Raw vs. FDS processed Data samples\nFeature Number: {df_raw.shape[1]+1}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    print("[Scatter plot] generated")
    print("[SCatter plot] PCA 1: ", explained_variance_ratio[0])
    print("[SCatter plot] PCA 2: ", explained_variance_ratio[1])
    plt.subplots_adjust(bottom=0.22)
    plt.text(0.01, -0.3, f'PC1 variance: {round(explained_variance_ratio[0],6)}; PC2 variance: {round(explained_variance_ratio[1],6)}\n[MSE] Raw: {round(results[0],0)}; Processed: {round(results[1],0)}; Reduced: {round(results[2],2)}%\n[PCC] Raw: {round(results[3],6)}; Processed: {round(results[4],6)}; Increased: {round(results[5],2)}%\n[NDCG] Raw: {round(results[6],6)}; Processed: {round(results[7],6)}; Increased: {round(results[8],2)}%', transform=plt.gca().transAxes, fontsize=10)
    plt.show()
    
    
    
def kmeans_plot(raw_data, smoothed_data, raw, smoothed, raw_df, smoothed_df, raw_ori_pca, smoothed_ori_pca):
    # Create figure and axes for the subplots
    fig, subplt = plt.subplots(2, figsize=(12, 6))  # 1 row, 2 columns
    raw_explained_variance_ratio = raw_ori_pca.explained_variance_ratio_
    sm_explained_variance_ratio = smoothed_ori_pca.explained_variance_ratio_
    # Plotting the RAW dataset
    subplt[0].scatter(raw_df[:, 0], raw_df[:, 1], c=raw.labels_, s=50, cmap='viridis')
    subplt[0].scatter(raw.cluster_centers_[:, 0], raw.cluster_centers_[:, 1], c='red', s=200, alpha=0.5, marker='X')
    subplt[0].set_title(f'RAW: K-Means Clustering with PCA Reduction\nFeature Number:{raw_data.shape[1]}')
    subplt[0].set_xlabel('PCA Feature 1')
    subplt[0].set_ylabel('PCA Feature 2')

    # Plotting the FDS dataset
    subplt[1].scatter(smoothed_df[:, 0], smoothed_df[:, 1], c=smoothed.labels_, s=50, cmap='viridis')
    subplt[1].scatter(smoothed.cluster_centers_[:, 0], smoothed.cluster_centers_[:, 1], c='red', s=200, alpha=0.5, marker='X')
    subplt[1].set_title(f'FDS SMOOTHED: K-Means Clustering with PCA Reduction\nFeature Number:{smoothed_data.shape[1]}')
    subplt[1].set_xlabel('PCA Feature 1')
    subplt[1].set_ylabel('PCA Feature 2')

    plt.text(0.01, -0.55, f'RAW: PC1 variance: {round(raw_explained_variance_ratio[0],6)}; PC2 variance: {round(raw_explained_variance_ratio[1],6)}\nFDS Processed: PC1 variance: {round(sm_explained_variance_ratio[0],6)}; PC2 variance: {round(sm_explained_variance_ratio[1],6)}\n[Inertia] Raw: {round(raw.inertia_,2)}; FDS processed: {round(smoothed.inertia_,2)}; Reduced: {round((raw.inertia_ - smoothed.inertia_ ) / raw.inertia_ *100, 2)}%', transform=plt.gca().transAxes, fontsize=10)


    plt.tight_layout()  # Adjusts subplot params so that the subplot(s) fits in to the figure area.
    plt.show()