import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ETL Pipeline Script

def extract_data(data_source=None):
    """Extracts data from a given source. If no source is provided, a sample dataset is used."""
    if data_source is None:
        # Sample data
        data = {
            'Name': ['Alice', 'Bob', 'Charlie', None],
            'Age': [25, None, 35, 40],
            'Gender': ['Female', 'Male', 'Male', None],
            'Salary': [50000, 60000, None, 80000]
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(data_source)
    print("Data Extracted:\n", df)
    return df

def create_pipeline():
    """Creates a preprocessing pipeline for numeric and categorical data."""
    numeric_features = ['Age', 'Salary']
    categorical_features = ['Gender']

    # Define numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Define categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def transform_data(df, pipeline):
    """Applies transformations to the data."""
    processed_data = pipeline.fit_transform(df)

    # Get feature names
    numeric_features = ['Age', 'Salary']
    categorical_features = pipeline.named_transformers_['cat']['onehot'].get_feature_names_out(['Gender'])
    all_features = numeric_features + list(categorical_features)

    # Convert to DataFrame
    processed_df = pd.DataFrame(processed_data, columns=all_features)
    print("Data Transformed:\n", processed_df)
    return processed_df

def load_data(df, target_file="processed_data.csv"):
    """Loads the processed data to a target file."""
    df.to_csv(target_file, index=False)
    print(f"Data Loaded to {target_file}")

# Main ETL Process
if __name__ == "__main__":
    # Step 1: Extract
    raw_data = extract_data()

    # Step 2: Create Pipeline
    preprocessing_pipeline = create_pipeline()

    # Step 3: Transform
    processed_data = transform_data(raw_data, preprocessing_pipeline)

    # Step 4: Load
    load_data(processed_data)
