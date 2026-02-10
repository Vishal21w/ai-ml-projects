from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models
ocean_model = joblib.load(os.path.join(BASE_DIR, "saved_models", "ocean_health_model.pkl"))
fish_model = joblib.load(os.path.join(BASE_DIR, "saved_models", "fisheries_yield_model.pkl"))
fish_features = joblib.load(os.path.join(BASE_DIR, "saved_models", "fisheries_features.pkl"))

HTML = """
<h2>AI Marine Data Platform</h2>

<h3>🌊 Ocean Health Prediction</h3>
<form method="post">
SST: <input name="sst" required><br>
Salinity: <input name="salinity" required><br>
<button name="type" value="ocean">Predict</button>
</form>

<h3>🐟 Fisheries Yield Prediction</h3>
<form method="post">
Latitude: <input name="lat" required><br>
Longitude: <input name="lon" required><br>
Water Temp: <input name="temp" required><br>
Species:
<select name="species">
<option>Sardine</option>
<option>Tuna</option>
<option>Mackerel</option>
</select><br>
<button name="type" value="fish">Predict</button>
</form>

<h3>{{ result }}</h3>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        if request.form["type"] == "ocean":
            df = pd.DataFrame([{
                "sst": float(request.form["sst"]),
                "salinity": float(request.form["salinity"])
            }])
            pred = ocean_model.predict(df)[0]
            result = f"Predicted Chlorophyll Level: {pred:.2f}"

        elif request.form["type"] == "fish":
            df = pd.DataFrame([{
                "latitude": float(request.form["lat"]),
                "longitude": float(request.form["lon"]),
                "water_temp": float(request.form["temp"]),
                "species": request.form["species"]
            }])

            df = pd.get_dummies(df)

            for col in fish_features:
                if col not in df:
                    df[col] = 0

            df = df[fish_features]
            pred = fish_model.predict(df)[0]
            result = f"Predicted Fish Catch: {pred:.2f} kg"

    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    app.run(debug=True)
