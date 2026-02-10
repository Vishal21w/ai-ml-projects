import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "processed", "ocean_cleaned.csv")
model_path = os.path.join(BASE_DIR, "saved_models", "ocean_health_model.pkl")

df = pd.read_csv(data_path)

X = df[['sst', 'salinity']]
y = df['chlorophyll']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, model_path)

y_pred = model.predict(X_test)

print("Ocean Health Model Saved Successfully")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
