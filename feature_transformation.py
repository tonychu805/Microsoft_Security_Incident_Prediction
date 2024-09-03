from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

def target_encode(df, n_bins):
    """
    Preprocesses the data by performing target encoding, frequency encoding, 
    and hashing for high cardinality columns.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame to preprocess.
    - low_cardinality_columns (list): List of columns with low cardinality for frequency encoding.
    - high_cardinality_columns (list): List of columns with high cardinality for hashing.
    - target_column (str): The name of the target variable for target encoding.
    - n_bins (int, optional): Number of bins for hashing high cardinality columns. Default is 100.
    
    Returns:
    - pd.DataFrame: The preprocessed DataFrame.
    """
    
    low_cardinality_columns = df.nunique()[df.nunique() <= 10].index
    high_cardinality_columns = df.drop(columns = ["Timestamp"]).nunique()[df.nunique() > 10].index

    # Copy the DataFrame to avoid altering the original data
    df_encoded = df.copy()
    
    # Drop the 'Timestamp' column
    if 'Timestamp' in df_encoded.columns:
        df_encoded = df_encoded.drop('Timestamp', axis=1)
    
    # Target encoding
    encoder = TargetEncoder()
    
    # Perform frequency encoding for low cardinality columns
    for column in low_cardinality_columns:
        df_encoded[column] = df_encoded[column].map(df_encoded[column].value_counts())
    
    # Perform hashing for high cardinality columns
    for column in high_cardinality_columns:
        df_encoded[column] = df_encoded[column].apply(lambda x: np.abs(hash(str(x))) % n_bins)
    
    return df_encoded


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled