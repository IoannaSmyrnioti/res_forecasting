import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_scada(df, timestamp_col="date_and_time", time_index=True, plot_sample=5000, correlation_threshold=0.95, output_dir="eda_outputs"):
    """
    General-purpose SCADA data EDA function.
    
    Parameters:
        df (pd.DataFrame): Input SCADA data.
        timestamp_col (str or None): Column name for timestamp. If None, assumes index.
        time_index (bool): Whether to set the timestamp column as index.
        plot_sample (int): Max points to plot in scatter/time series.
    """
    df = df.copy()  # <-- για να μην αλλάζεις το αρχικό df
    os.makedirs(output_dir, exist_ok=True)  # <-- για τα savefig

    print("Dataset Overview")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    
    # Dtypes and nulls
    print("\n Dtypes and Missing Values:")
    print(df.info())
    print(df.dtypes) 
    print(df.isna().sum()[df.isna().sum() > 0])

    # Parse datetime and sort index
    if timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        if time_index:
            df.set_index(timestamp_col, inplace=True)
        df.sort_index(inplace=True)

    print("After index set, df is of type:", type(df))  

    print("\n Time Range:", df.index.min(), "to", df.index.max())
    print("Duplicate Timestamps:", df.index.duplicated().sum())

    # --- Plot time series
    df_numeric = df.select_dtypes(include=[np.number])

    print("\n Plotting Time Series (numeric features)...")
    df_numeric.iloc[:plot_sample].plot(subplots=True, figsize=(15, min(3*len(df_numeric.columns), 20)))
    plt.suptitle("SCADA Time Series Preview", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/TimeSeries_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --- Plot histograms
    df_numeric.hist(bins=30, figsize=(15, 10), layout=(min(4, len(df_numeric.columns)), -1))
    plt.suptitle("Histograms of SCADA Variables", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/histograms.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --- Plot boxplots
    df_numeric.plot(kind='box', subplots=True, layout=(min(4, len(df_numeric.columns)), -1), figsize=(15, 10))
    plt.suptitle("Boxplots for Outlier Detection", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/boxplots_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Correlation heatmap
    corr = df_numeric.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Feature redundancy detection (based on correlation)
    def find_redundant_features(corr_matrix, threshold):
        corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)
        corr_pairs = corr_pairs[corr_pairs < 1.0]  # remove self-pairs
        redundant = corr_pairs[corr_pairs > threshold].drop_duplicates()
        return redundant

    redundant_pairs = find_redundant_features(corr, threshold=correlation_threshold)

    print("\n Highly Correlated Feature Pairs (|r| > {correlation_threshold}):")
    if redundant_pairs.empty:
        print("No strongly redundant pairs detected.")
    else:
        print(redundant_pairs)
            
    return {
        "correlation_matrix": corr,
        "redundant_pairs": redundant_pairs
    }

def explore_and_save(df, output_dir="eda_outputs", correlation_threshold=0.95):
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = explore_scada(
        df,
        timestamp_col="date_and_time",
        time_index=True,
        plot_sample=5000,
        correlation_threshold=correlation_threshold,
        output_dir=output_dir
    )

    # Save CSV outputs
    results["correlation_matrix"].to_csv(f"{output_dir}/correlation_matrix.csv")
    if not results["redundant_pairs"].empty:
        results["redundant_pairs"].to_csv(f"{output_dir}/highly_correlated_features.csv")

    print(f"\n EDA results saved in: {output_dir}/")

    return results


