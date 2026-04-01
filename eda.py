import pandas as pd

def perform_eda(df):
    report = {
        "shape": df.shape,
        "missing": df.isnull().sum().to_dict(),
        "summary": df.describe().to_dict()
    }
    
    # Simple Leakage/Correlation check
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr().abs()
        # Find features highly correlated with others
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr.append(f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}")
        report["high_correlation_pairs"] = high_corr
        
    return report