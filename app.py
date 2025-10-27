from flask import Flask, request, jsonify
import pickle
import pandas as pd
import requests
import os

app = Flask(__name__)

OPENWEATHER_KEY = "d37c60c3683f05a914ca03094014a96f"
AGROMONITOR_KEY = "260b98399f72f91d1f217375444447f8f"

MODEL_PATH = 'crop_yield_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_KEY}&units=metric"
        res = requests.get(url).json()
        temp = res["main"]["temp"]
        humidity = res["main"]["humidity"]
        rainfall = res.get("rain", {}).get("1h", 0)
        return temp, humidity, rainfall
    except Exception as e:
        print(f"Weather fetch error: {e}")
        return None, None, None


def get_soil(lat, lon):
    try:
        url = f"https://api.agromonitoring.com/agro/1.0/soil?lat={lat}&lon={lon}&appid={AGROMONITOR_KEY}"
        res = requests.get(url).json()
        moisture = res.get("moisture", 0)
        soil_temp = res.get("t0", 0) - 273.15 if res.get("t0") else None
        return {"soil_moisture": moisture, "soil_temp": soil_temp}
    except Exception as e:
        print(f"Soil fetch error: {e}")
        return {"soil_moisture": None, "soil_temp": None}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        city = data.get("city")
        lat, lon = data.get("lat"), data.get("lon")

        temp, humidity, rainfall = get_weather(city)
        soil_data = get_soil(lat, lon)

       
        df = pd.DataFrame([{
            "Soil_Type": data["Soil_Type"],
            "Crop_Type": data["Crop_Type"],
            "Season": data["Season"],
            "pH": data["pH"],
            "Nitrogen": data["Nitrogen"],
            "Phosphorus": data["Phosphorus"],
            "Potassium": data["Potassium"],
            "Rainfall": rainfall if rainfall else 100,
            "Temperature": temp if temp else 25,
            "Humidity": humidity if humidity else 60,
            "Irrigation_Type": data["Irrigation_Type"]
        }])

        pred = model.predict(df)[0]

        return jsonify({
            "city": city,
            "temperature": temp,
            "humidity": humidity,
            "rainfall": rainfall,
            "soil_conditions": soil_data,
            "predicted_yield": round(float(pred), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "🌾 Crop Yield Predictor API is running.",
        "endpoints": {
            "POST /predict": "Predict crop yield based on soil, weather & location."
        }
    })


if __name__ == "__main__":
    print("🚀 Starting Crop Yield Predictor API...")
    app.run(host="0.0.0.0", port=5000, debug=True)
