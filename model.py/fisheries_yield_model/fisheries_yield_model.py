import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "processed", "fisheries_cleaned.csv")

model_path = os.path.join(BASE_DIR, "saved_models", "fisheries_yield_model.pkl")
features_path = os.path.join(BASE_DIR, "saved_models", "fisheries_features.pkl")

df = pd.read_csv(data_path)

# Encode species
df_encoded = pd.get_dummies(df, columns=['species'])

X = df_encoded.drop(columns=['date', 'catch_kg'])
y = df_encoded['catch_kg']

# Save feature names
joblib.dump(X.columns.tolist(), features_path)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, model_path)

print("Fisheries model + feature list saved successfully")
