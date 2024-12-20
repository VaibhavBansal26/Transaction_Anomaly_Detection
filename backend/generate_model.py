import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
import joblib

# Function to load data
def load_data(filepath):
    return pd.read_csv(filepath)

# Custom preprocessing function
def preprocess_data(data):
    # Handle missing values
    data.dropna(inplace=True)

    # Select relevant features
    relevant_features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']

    # Check if all relevant features exist
    if not all(feature in data.columns for feature in relevant_features):
        raise ValueError("Dataset is missing required columns")

    # Add target column (example anomaly flag logic)
    mean_amount = data['Transaction_Amount'].mean()
    std_amount = data['Transaction_Amount'].std()
    anomaly_threshold = mean_amount + 2 * std_amount
    data['Is_Anomaly'] = data['Transaction_Amount'] > anomaly_threshold

    return data

# Function to split data into training and testing sets
def split_data(data, features, target_column):
    X = data[features]
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train the model using a pipeline and save it
def train_and_save_pipeline(data, model_path):
    # Define the pipeline
    pipeline = Pipeline([
        ('preprocessor', FunctionTransformer(preprocess_data, validate=False)),
        ('scaler', StandardScaler()),
        ('model', IsolationForest(contamination=0.02, random_state=42))
    ])

    # Relevant features for the pipeline
    relevant_features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']

    # Prepare the data
    X = data[relevant_features]

    # Train the pipeline
    pipeline.fit(X)

    # Save the pipeline
    joblib.dump(pipeline, model_path)
    print(f"Pipeline saved to {model_path}")

# Main script
if __name__ == "__main__":
    filepath = "transaction_anomalies_dataset.csv"  # Update with the actual dataset path
    model_path = "rfr_v1.pkl"  # Path to save the pipeline

    # Load data
    data = load_data(filepath)

    # Train and save the pipeline
    train_and_save_pipeline(data, model_path)

    print("Pipeline completed. Model is ready for API integration.")