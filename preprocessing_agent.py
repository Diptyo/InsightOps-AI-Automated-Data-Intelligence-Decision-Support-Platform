import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class PreprocessingAgent:
    def __init__(self):
        self.label_encoders = {}

    def process(self, df: pd.DataFrame, target_col=None):
        df = df.copy()
        print(f"🛠️  Agent: Analyzing {df.shape[1]} columns for universal optimization...")
        
        # 1. REMOVE DUPLICATES & EMPTY COLUMNS
        df = df.drop_duplicates()
        df = df.dropna(axis=1, how='all')

        # 2. AGNOSTIC "NOISE" CLEANING (The String-to-Float Fix)
        # This replaces any string that looks like a placeholder with NaN
        for col in df.columns:
            if df[col].dtype == 'object':
                # Identify if a specific string takes up more than 5% of the data 
                # and contains non-alphabetic characters (like 9999-99-99 or 999)
                value_counts = df[col].value_counts(normalize=True)
                for val, percent in value_counts.items():
                    if isinstance(val, str) and any(char.isdigit() for char in val):
                        if "-" in val or "/" in val or val.isdigit():
                            # If a weird number-string is very common, it's likely a placeholder
                            if percent > 0.05: 
                                df[col] = df[col].replace(val, np.nan)

        # 3. DYNAMIC DATE DETECTION
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Attempt conversion now that noise is removed
                    temp_date = pd.to_datetime(df[col], errors='coerce')
                    if temp_date.notnull().sum() / len(df) > 0.4: 
                        df[f'{col}_year'] = temp_date.dt.year.fillna(-1)
                        df[f'{col}_month'] = temp_date.dt.month.fillna(-1)
                        df[f'{col}_day'] = temp_date.dt.day.fillna(-1)
                        df = df.drop(columns=[col])
                except:
                    continue

        # 4. TYPE-BASED CLEANING 
        for col in df.columns:
            if col == target_col: continue

            if df[col].dtype == 'object':
                numeric_attempt = pd.to_numeric(df[col], errors='coerce')
                # If it's mostly numbers now, convert it
                if numeric_attempt.notnull().sum() / len(df) > 0.5:
                    df[col] = numeric_attempt
                else:
                    df[col] = df[col].astype('category')

        # 5. SMART IMPUTATION
        num_cols = df.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        cat_cols = df.select_dtypes(include=['category', 'object']).columns
        if not cat_cols.empty:
            df[cat_cols] = df[cat_cols].fillna('Unknown')

        # 6. TARGET ENCODING (Classification vs Regression)
        if target_col and target_col in df.columns:
            # Check if target is categorical (string or few unique integers)
            unique_vals = df[target_col].nunique()
            if df[target_col].dtype == 'object' or unique_vals < 15:
                le = LabelEncoder()
                df[target_col] = le.fit_transform(df[target_col].astype(str))
                self.label_encoders[target_col] = le

        # 7. CATEGORICAL OPTIMIZATION
        for col in df.select_dtypes(include=['object']).columns:
            if col != target_col:
                df[col] = df[col].astype('category')

        # 8. ADD THIS: NUMERICAL ENCODING FOR MODELS
        # This converts categories into 1s and 0s (One-Hot Encoding)
        # Standard models require this to avoid "Cannot cast str to float" errors.
        categorical_features = df.select_dtypes(include=['category']).columns
        if not categorical_features.empty:
            df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

        # 🔥 THE ABSOLUTE FIX: You must return the processed DataFrame
        return df