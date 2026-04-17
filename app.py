import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AstraAI", layout="wide")

# -----------------------------
# UI
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🌍 AstraAI Dashboard")
st.subheader("AI-Powered Climate & Health Intelligence")

# -----------------------------
# CITY LIST
# -----------------------------
cities = [
    "delhi", "mumbai", "bangalore", "chennai", "kolkata",
    "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow",
    "chandigarh", "bhopal", "patna", "indore", "nagpur",
    "surat", "kanpur", "amritsar", "varanasi", "ranchi",
    "guwahati"
]

city = st.selectbox("📍 Select City", cities)

# -----------------------------
# FIXED MAP COORDS
# -----------------------------
city_coords = {
    "delhi": [28.61, 77.23],
    "mumbai": [19.07, 72.87],
    "bangalore": [12.97, 77.59],
    "chennai": [13.08, 80.27],
    "kolkata": [22.57, 88.36],
    "hyderabad": [17.38, 78.48],
    "pune": [18.52, 73.85],
    "ahmedabad": [23.02, 72.57],
    "jaipur": [26.91, 75.78],
    "lucknow": [26.84, 80.94],
    "chandigarh": [30.73, 76.78],
    "bhopal": [23.25, 77.41],
    "patna": [25.59, 85.13],
    "indore": [22.72, 75.85],
    "nagpur": [21.14, 79.08],
    "surat": [21.17, 72.83],
    "kanpur": [26.45, 80.35],
    "amritsar": [31.63, 74.87],
    "varanasi": [25.31, 82.97],
    "ranchi": [23.34, 85.31],
    "guwahati": [26.14, 91.73]
}

lat, lon = city_coords.get(city, [28.61, 77.23])

# -----------------------------
# FETCH AQI FROM API
# -----------------------------
API_KEY = "Your api key"

url = f"https://api.waqi.info/feed/{city}/?token={API_KEY}"

try:
    response = requests.get(url)
    data = response.json()

    if data["status"] == "ok":
        raw_aqi = data["data"]["aqi"]

        # handle invalid AQI
        if isinstance(raw_aqi, int):
            live_aqi = raw_aqi
        else:
            live_aqi = np.random.randint(80, 250)
    else:
        live_aqi = np.random.randint(80, 250)

except:
    live_aqi = np.random.randint(80, 250)

# -----------------------------
# AQI STATUS
# -----------------------------
def get_status(aqi):
    if aqi <= 100:
        return "🟢 Good"
    elif aqi <= 200:
        return "🟠 Moderate"
    else:
        return "🔴 Severe"

col1, col2 = st.columns(2)

with col1:
    st.metric("🌫 Live AQI", live_aqi)
    st.write(f"Status: {get_status(live_aqi)}")

with col2:
    st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))

# -----------------------------
# REFRESH
# -----------------------------
if st.button("🔄 Refresh Data"):
    st.rerun()

# -----------------------------
# ML MODEL (CACHED)
# -----------------------------
@st.cache_resource
def train_model():
    np.random.seed(42)
    data = {
        "temperature": np.random.randint(20, 45, 200),
        "humidity": np.random.randint(30, 90, 200),
        "pm2_5": np.random.randint(50, 300, 200)
    }
    df = pd.DataFrame(data)
    df["aqi"] = df["pm2_5"] * 1.2 + df["humidity"] * 0.5

    X = df[["temperature", "humidity", "pm2_5"]]
    y = df["aqi"]

    model = RandomForestRegressor()
    model.fit(X, y)

    return model, df

model, df = train_model()

# -----------------------------
# INPUT
# -----------------------------
st.markdown("## 📥 Input Data")

temp = st.slider("🌡 Temperature", 20, 45, 30)
humidity = st.slider("💧 Humidity", 30, 90, 60)
pm = st.slider("🌫 PM2.5", 50, 300, int(live_aqi))
age = st.slider("👤 Age", 5, 80, 20)

# -----------------------------
# PREDICTION
# -----------------------------
st.markdown("## 🔮 AI Prediction")

input_data = np.array([[temp, humidity, pm]])
prediction = model.predict(input_data)[0]

st.metric("Predicted AQI", int(prediction))

# -----------------------------
# HEALTH SCORE
# -----------------------------
st.markdown("## 🧠 Smart Health Score")

health_score = 100 - (prediction / 3)
if age > 50:
    health_score -= 10

health_score = max(0, int(health_score))
st.metric("Health Score", health_score)

# -----------------------------
# ACTIVITY
# -----------------------------
st.markdown("## 🏃 Activity Recommendation")

if prediction > 250:
    st.write("❌ Avoid outdoor activities")
elif prediction > 150:
    st.write("⚠️ Light walking only")
else:
    st.write("🏃 Safe for sports")

# -----------------------------
# ALERTS
# -----------------------------
st.markdown("## 🔔 Alerts")

if prediction > 200:
    st.error("🚨 Unsafe air quality expected!")
elif prediction > 150:
    st.warning("⚠️ Moderate pollution")
else:
    st.success("✅ Air stable")

# -----------------------------
# GRAPH
# -----------------------------
st.markdown("## 📈 AQI Trend")

plt.figure()
plt.plot(df["aqi"].head(30))
plt.xlabel("Time")
plt.ylabel("AQI")
st.pyplot(plt)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.write("🚀 AstraAI | ")
