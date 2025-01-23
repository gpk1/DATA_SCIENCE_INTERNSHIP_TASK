import pandas as pd
from sklearn.datasets import make_classification

# Generate synthetic data
def generate_data():
    X, y = make_classification(
        n_samples=1000,  # 1000 samples
        n_features=20,  # 20 features
        n_informative=15,  # 15 informative features
        n_redundant=5,  # 5 redundant features
        random_state=42
    )
    columns = [f'feature_{i}' for i in range(1, 21)]  # Column names for features
    df = pd.DataFrame(X, columns=columns)  # Features
    df['target'] = y  # Add target column (customer churn: 0 or 1)
    return df

# Save the data to a CSV file
df = generate_data()
df.to_csv('customer_churn_data.csv', index=False)
print("Data saved to 'customer_churn_data.csv'")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('customer_churn_data.csv')

# Split data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler using joblib
import joblib
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('customer_churn_data.csv')

# Split data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler using joblib
import joblib
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save the trained model
joblib.dump(model, 'model.pkl')
print("Model saved as 'model.pkl'")

# Print accuracy and classification report
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
