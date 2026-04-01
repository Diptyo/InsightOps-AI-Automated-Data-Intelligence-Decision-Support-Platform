"""
Inspect dataset structure and metadata.
"""

def inspect_dataset(df):

    info = {}

    info["rows"] = df.shape[0]
    info["columns"] = df.shape[1]
    info["column_names"] = list(df.columns)

    info["missing_values"] = df.isnull().sum().to_dict()

    return info