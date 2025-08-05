# train_model.py (REVISED)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Starting model training and saving process (4 features)...")

# Load the dataset
try:
    df = pd.read_csv('parkinsons.data')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: parkinsons.data not found. Make sure it's in the same directory.")
    exit()

# --- IMPORTANT: Select ONLY the 4 desired features here ---
selected_features = ['PPE', 'spread1', 'spread2', 'MDVP:Fo(Hz)'] # <--- YOUR CHOSEN 4 FEATURES

# Ensure all selected features are in the dataframe
if not all(feature in df.columns for feature in selected_features):
    print("ERROR: One or more selected features not found in the dataset.")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

X = df[selected_features] # <--- ONLY THESE FEATURES ARE USED NOW
y = df['status']

print(f"Using {len(selected_features)} features for training: {selected_features}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data split into training and testing sets.")

# Initialize and fit the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaler fitted and data scaled.")

# Convert scaled arrays back to DataFrames to maintain column names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=selected_features) # Use selected_features for columns
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=selected_features)

# Train the Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled_df, y_train)
print("Random Forest model trained.")

# Save the trained model and scaler
joblib.dump(rf_model, 'parkinsons_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully as 'parkinsons_rf_model.pkl' and 'scaler.pkl'.")
print("Training complete.")