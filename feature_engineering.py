import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df, target):
    """
    Memory-efficient generic preprocessing.
    Works for both Regression and Classification.
    """
    print("🛠️  Running Feature Engineering...")

    # --- 1. HANDLE STRINGS, DATES, & CATEGORIES ---
    for col in df.columns:
        if col == target:
            continue

        # FIX: Use pandas api instead of np.issubdtype
        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        if is_numeric:
            df[col] = df[col].fillna(df[col].median())
            continue

        # Handle Dates
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f"{col}_year"] = df[col].dt.year.fillna(0).astype(np.int32)
            df[f"{col}_month"] = df[col].dt.month.fillna(0).astype(np.int8)
            df.drop(columns=[col], inplace=True)
            continue

        # Handle Categorical/Strings
        df[col] = df[col].fillna("Unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str)).astype(np.int32)

    # --- 2. TARGET HANDLING ---
    # FIX: Use pandas api check here too
    if not pd.api.types.is_numeric_dtype(df[target]):
        print(f"Label Encoding categorical target: {target}")
        le_target = LabelEncoder()
        df[target] = le_target.fit_transform(df[target])
    else:
        print(f"Target {target} is numeric. Skipping encoding.")

    # --- 3. FEATURE SCALING (Memory Efficient) ---
    features = [c for c in df.columns if c != target]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    print("✅ Feature Engineering complete.")
    return df