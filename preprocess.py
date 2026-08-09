import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_and_preprocess_data(csv_path='flights.csv'):
    """
    Loads flight data from a CSV, performs a safe train-test split, 
    and executes hybrid encoding using robust scikit-learn components.
    """
    # 1. Load Data from CSV (low_memory=False stops dtype warnings)
    df_flights = pd.read_csv(csv_path, low_memory=False)
    
    # 2. Define Features and Target
    features = ['SCHEDULED_DEPARTURE', 'MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
    
    X = df_flights[features].copy()
    y = df_flights['ARRIVAL_DELAY'].copy()

    # Drop rows where target or critical features have missing values to avoid NaN issues
    df_combined = X.copy()
    df_combined['ARRIVAL_DELAY'] = y
    df_combined = df_combined.dropna()
    
    X = df_combined[features].copy()
    y = df_combined['ARRIVAL_DELAY'].copy()

    # 3. Train/Test Split FIRST (Crucial to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create working copy of training data to safely calculate target encoding statistics
    train_df = X_train.copy()
    train_df['ARRIVAL_DELAY'] = y_train

    # 4. Numerical Features
    df_num_train = train_df[['SCHEDULED_DEPARTURE']]
    df_num_test = X_test[['SCHEDULED_DEPARTURE']]

    # 5. Categorical Features Processing
    cat_cols = ['MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
    df_cat_train = train_df[cat_cols].astype(str)
    df_cat_test = X_test[cat_cols].astype(str)

    # Target Encoding for high-cardinality airports (computed EXCLUSIVELY on training data)
    origin_means = train_df.groupby('ORIGIN_AIRPORT')['ARRIVAL_DELAY'].mean()
    dest_means = train_df.groupby('DESTINATION_AIRPORT')['ARRIVAL_DELAY'].mean()

    df_cat_train['origin_encoded'] = df_cat_train['ORIGIN_AIRPORT'].map(origin_means)
    df_cat_test['origin_encoded'] = df_cat_test['ORIGIN_AIRPORT'].map(origin_means)

    df_cat_train['dest_encoded'] = df_cat_train['DESTINATION_AIRPORT'].map(dest_means)
    df_cat_test['dest_encoded'] = df_cat_test['DESTINATION_AIRPORT'].map(dest_means)

    # Handle unseen categories in the test set by filling with the training global mean
    global_mean = y_train.mean()
    df_cat_test['origin_encoded'] = df_cat_test['origin_encoded'].fillna(global_mean)
    df_cat_test['dest_encoded'] = df_cat_test['dest_encoded'].fillna(global_mean)

    # Robust One-Hot Encoding for low-cardinality features (updated sparse_output for newer sklearn)
    low_card_cols = ['MONTH', 'DAY_OF_WEEK', 'AIRLINE']
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    
    dummy_train_arr = encoder.fit_transform(df_cat_train[low_card_cols])
    dummy_test_arr = encoder.transform(df_cat_test[low_card_cols])
    
    encoded_col_names = encoder.get_feature_names_out(low_card_cols)
    
    dummy_train = pd.DataFrame(dummy_train_arr, columns=encoded_col_names, index=df_cat_train.index)
    dummy_test = pd.DataFrame(dummy_test_arr, columns=encoded_col_names, index=df_cat_test.index)

    # 6. Concatenate Everything Sideways
    X_train_final = pd.concat([df_num_train, dummy_train, df_cat_train[['origin_encoded', 'dest_encoded']]], axis=1).astype(float)
    X_test_final = pd.concat([df_num_test, dummy_test, df_cat_test[['origin_encoded', 'dest_encoded']]], axis=1).astype(float)

    return X_train_final, X_test_final, y_train, y_test