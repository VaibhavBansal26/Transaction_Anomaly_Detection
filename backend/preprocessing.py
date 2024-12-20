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
