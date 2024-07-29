import time
import datetime
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import requests
from fpdf import FPDF
import ee
import geemap
import os

# --- Configuration ---
WEATHERBIT_API_KEY = "149c81a755124c8c99d4382c37fca1e1"  # Your Weatherbit API key
AIRNOW_API_KEY = "FAE7D497-D333-43D5-807B-776063413223"
OPENAQ_API_KEY = "36a6c5799795c35341961080ac0dd940a5b9899dae77036a476689fa03be3a60"
YOUR_OPENWEATHERMAP_API_KEY="7f35e12a8024ed949957c218f044a958"
CO2_THRESHOLD = 600
NOISE_THRESHOLD = 60
IDEAL_TEMP_RANGE = (20, 24)
DATA_CSV = 'environmental_data.csv'
REPORT_FOLDER = 'reports'

# Create the reports directory if it doesn't exist
if not os.path.exists(REPORT_FOLDER):
    os.makedirs(REPORT_FOLDER)

# --- Data Collection (Simulated) ---
def collect_data():
    data = []
    for i in range(100):
        timestamp = datetime.datetime.now() + datetime.timedelta(minutes=i)
        temperature = np.random.uniform(18, 25)
        humidity = np.random.uniform(40, 60)
        co2 = np.random.uniform(400, 600)
        light_intensity = np.random.randint(100, 500)
        noise_level = np.random.randint(30, 70)
        voc = np.random.uniform(0, 1)  
        data.append([timestamp, temperature, humidity, co2, light_intensity, noise_level, voc])
    return pd.DataFrame(data, columns=['timestamp', 'temperature', 'humidity', 'co2',
                                       'light_intensity', 'noise_level', 'voc'])

# --- Data Processing ---
def process_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    scaler = MinMaxScaler()
    df[['temperature', 'humidity', 'co2', 'light_intensity', 'noise_level', 'voc']] = \
        scaler.fit_transform(df[['temperature', 'humidity', 'co2', 'light_intensity', 'noise_level', 'voc']])
    df['relative_humidity'] = df['humidity'] / df['temperature']
    return df


# --- Data Integration (Weatherbit) ---
def integrate_weather_data(df, location="Ahmedabad,IN"):
    base_url = "https://api.weatherbit.io/v2.0/current"
    complete_url = f"{base_url}?lat=23.0225&lon=72.5714&key={WEATHERBIT_API_KEY}&include=minutely"
    response = requests.get(complete_url)
    data = response.json()
    if response.status_code == 200:
        try:
            df['outside_temperature'] = data['data'][0]['temp']
            df['wind_speed'] = data['data'][0]['wind_spd']
            df['cloud_coverage'] = data['data'][0]['clouds']
            df['precipitation'] = data['data'][0]['precip']  
            return df
        except KeyError as e:
            print(f"Error extracting weather data: {e}")
            print(data)
            return df
    else:
        print(f"Error: Weather API request failed: {response.status_code}")
        return df


# --- Data Integration (AirNow) ---
def integrate_air_quality_data(df, zip_code="90001"):
    url = f"https://www.airnowapi.org/aq/observation/zipCode/current/?format=application/json&zipCode={zip_code}&distance=25&API_KEY={AIRNOW_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        try:
            aqi_data = {}
            for item in data:
                parameter = item['ParameterName']
                aqi = item.get('AQI')
                if aqi is not None:
                    aqi_data[parameter] = aqi

            if aqi_data:
                if 'OZONE' in aqi_data:
                    df['air_quality_index'] = aqi_data['OZONE']
                elif 'PM2.5' in aqi_data:
                    df['air_quality_index'] = aqi_data['PM2.5']
                else:
                    df['air_quality_index'] = max(aqi_data.values())
            else:
                df['air_quality_index'] = np.nan
            return df
        except Exception as e:
            print(f"Error extracting AQI from AirNow data: {e}")
            print(data)
            return df
    else:
        print(f"Error fetching air quality data: {response.status_code}")
        return df

def integrate_openaq_data(df, location="Ahmedabad"):
    """Fetches air quality data from OpenAQ API."""
    base_url = "https://api.openaq.org/v2/latest"
    complete_url = f"{base_url}?limit=1&location={location}"  # Get latest data for location
    response = requests.get(complete_url)
    data = response.json()
    if response.status_code == 200:
        try:
            pm25_value = next(
                (item['value'] for item in data['results'][0]['measurements'] if item['parameter'] == 'pm25'),
                None
            )
            df['openaq_pm25'] = pm25_value if pm25_value is not None else np.nan
            return df
        except (IndexError, KeyError) as e:
            print(f"Error extracting PM2.5 from OpenAQ data: {e}")
            print(data)
            return df
    else:
        print(f"Error: OpenAQ API request failed: {response.status_code}")
        return df

# --- Satellite Image Monitoring ---
def monitor_satellite_images(df, roi_coordinates):
    roi = ee.Geometry.Point(roi_coordinates)

    # Filter based on your desired date range
    ndvi_images = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterBounds(roi) \
        .filterDate('2023-01-01', '2023-12-31') 

    deforestation_rate = analyze_deforestation(ndvi_images, roi)

    # Add the deforestation rate to the DataFrame
    df['deforestation_rate'] = deforestation_rate
    return df


# --- Deforestation Analysis ---
def analyze_deforestation(ndvi_images, roi):
    first_image = ndvi_images.first()
    last_image = ndvi_images.sort('system:time_start', False).first()

    first_ndvi = first_image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    last_ndvi = last_image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

    ndvi_diff = last_ndvi.subtract(first_ndvi).rename('NDVI_difference')

    deforestation_threshold = -0.1
    deforestation_mask = ndvi_diff.lt(deforestation_threshold)

    deforestation_area = deforestation_mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=30
    )

    total_area = roi.area().divide(10000)

    if total_area.getInfo() == 0:
        deforestation_rate = 0
    else:
        deforestation_rate = deforestation_area.get('NDVI_difference').getInfo() / total_area.getInfo()

    deforestation_map = ee.Image.constant(deforestation_rate).clip(roi)
    map = geemap.Map()
    
    # Add layers with clearer descriptions and color palettes
    map.addLayer(deforestation_map, {'palette': ['green', 'yellow', 'red']},
                 'Deforestation Rate (Green: No Deforestation, Yellow: Moderate, Red: High)')
    map.addLayer(ndvi_diff, {'min': -0.2, 'max': 0.2, 'palette': ['blue', 'white', 'red']}, 'NDVI Difference (Blue: Decreased Vegetation, Red: Increased Vegetation)')
    map.addLayer(roi, {}, 'Region of Interest (Ahmedabad)')

    # Center the map on Ahmedabad
    ahmedabad = ee.Geometry.Point([72.5714, 23.0225]) 
    map.centerObject(ahmedabad) 

    html = map.to_html()

    with open("deforestation_map.html", "w") as f:
        f.write(html)

    print("Deforestation map generated in deforestation_map.html. Please open the file in a web browser.")
    return deforestation_rate

def predict_air_quality(df, roi):
    """Predicts air quality using a linear regression model trained on IoT and satellite data."""

    # Load MODIS AOD data (adjust time range and collection as needed)
    aod_data = ee.ImageCollection("MODIS/006/MOD04_3K") \
        .filterBounds(roi) \
        .filterDate('2023-01-01', '2023-12-31') \
        .select('Optical_Depth_047')

    # Extract AOD values for each timestamp in the IoT data
    aod_values = []
    for timestamp in df.index:
        aod_image = aod_data.filterDate(timestamp, timestamp + pd.Timedelta(minutes=1))
        if not aod_image.isEmpty():
            aod_value = aod_image.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi).get('Optical_Depth_047').getInfo()
            aod_values.append(aod_value)
        else:
            aod_values.append(np.nan)

    # Combine AOD with relevant IoT data (adjust features as needed)
    X = df[['temperature', 'humidity', 'co2', 'wind_speed']].values
    y = aod_values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Air Quality Prediction Model MSE: {mse}")

    # Predict air quality for the full dataset
    y_pred_all = model.predict(X)
    air_quality_predictions = pd.DataFrame({'predicted_aod': y_pred_all}, index=df.index)

    return air_quality_predictions

def generate_conservation_insights(df):
    """Generates conservation insights based on deforestation and air quality trends."""

    insights = []

    # Deforestation Insights
    if df['deforestation_rate'].mean() > 0.05:  # Adjust threshold as needed
        insights.append("High deforestation rate detected, indicating significant forest loss. Recommend urgent action.")
    elif df['deforestation_rate'].mean() > 0.01:
        insights.append("Moderate deforestation rate detected, suggesting ongoing forest degradation. Consider mitigation strategies.")

    # Air Quality Insights
    if df['predicted_aod'].mean() > 0.5:  # Adjust threshold as needed
        insights.append("High levels of air pollution predicted, potentially impacting health and the environment.")
        insights.append("Recommend implementing air quality management strategies and promoting cleaner energy sources.")
    elif df['predicted_aod'].mean() > 0.2:
        insights.append("Elevated levels of air pollution predicted, potentially causing health concerns.")
        insights.append("Promote green transportation, urban forestry, and sustainable industrial practices.")

    # Combined Insights
    if df['deforestation_rate'].mean() > 0.01 and df['predicted_aod'].mean() > 0.2:
        insights.append("Deforestation and air pollution are interconnected. Consider integrated conservation efforts.")

    return insights

# --- Model Development & Training ---
def develop_model(df):
    if df is None or len(df) < 2:
        print("Not enough data to train. Using a simple placeholder model.")
        return None
    features = ['temperature', 'humidity', 'light_intensity', 'noise_level', 'voc',
                'outside_temperature', 'wind_speed', 'cloud_coverage', 'precipitation']
    try:
        X = df[features]
        y = df['co2']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error during model training: {e}")
        return None


def train_model(model, df):
    if model is None:
        return develop_model(df)
    features = ['temperature', 'humidity', 'light_intensity', 'noise_level', 'voc',
                'outside_temperature', 'wind_speed', 'cloud_coverage', 'precipitation']
    X = df[features]
    y = df['co2']
    model.fit(X, y)
    return model


# --- Model Validation ---
def validate_model(model, df):
    if model is None:
        print("Model not available for validation.")
        return
    features = ['temperature', 'humidity', 'light_intensity', 'noise_level', 'voc',
                'outside_temperature', 'wind_speed', 'cloud_coverage', 'precipitation']
    X = df[features]
    y = df['co2']
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")


# --- Prediction ---
def make_predictions(model, df):
    if model is None:
        print("Model not available for prediction.")
        return None
    features = ['temperature', 'humidity', 'light_intensity', 'noise_level', 'voc',
                'outside_temperature', 'wind_speed', 'cloud_coverage', 'precipitation']
    X = df[features]
    y_pred = model.predict(X)
    return y_pred


# --- Monitoring ---
def monitor_data(df, co2_threshold=CO2_THRESHOLD, voc_threshold=0.5):  # Add VOC threshold
    co2_anomalies = df[df['co2'] > co2_threshold]
    voc_anomalies = df[df['voc'] > voc_threshold]  # Monitor VOC
    if not co2_anomalies.empty:
        print("High CO2 Levels Detected:")
        print(co2_anomalies)
    if not voc_anomalies.empty:
        print("High VOC Levels Detected:")
        print(voc_anomalies)


# --- Insights Generation ---
def generate_insights(df, model, co2_threshold=CO2_THRESHOLD):
    insights = []
    if model is None:
        insights.append("No model available. Insights are limited to basic data analysis.")
        return insights

    predictions = make_predictions(model, df)
    if predictions is None:
        return insights

    high_risk = df[predictions > co2_threshold]
    if not high_risk.empty:
        high_risk_times = high_risk.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        insights.append(
            f"Potential high CO2 risks (above {co2_threshold} ppm) detected at: {', '.join(high_risk_times)}")

    correlation_with_temp = df[['co2', 'outside_temperature']].corr().iloc[0, 1]
    if correlation_with_temp > 0.5:
        insights.append(f"Strong positive correlation ({correlation_with_temp:.2f}) "
                       f"found between CO2 and outside temperature.")
    elif correlation_with_temp < -0.5:
        insights.append(f"Strong negative correlation ({correlation_with_temp:.2f}) "
                       f"found between CO2 and outside temperature.")

    df['hour'] = df.index.hour
    hourly_co2 = df.groupby('hour')['co2'].mean()
    peak_hour = hourly_co2.idxmax()
    insights.append(f"Average CO2 levels peak around {peak_hour}:00 hours.")

    # Add insights for VOC
    correlation_with_voc = df[['co2', 'voc']].corr().iloc[0, 1]
    if correlation_with_voc > 0.5:
        insights.append(f"Strong positive correlation ({correlation_with_voc:.2f}) "
                       f"found between CO2 and VOC levels.")
    elif correlation_with_voc < -0.5:
        insights.append(f"Strong negative correlation ({correlation_with_voc:.2f}) "
                       f"found between CO2 and VOC levels.")

    return insights


# --- Occupancy Estimation ---
def estimate_occupancy(df, co2_baseline=400, co2_per_person=40):
    df['estimated_occupancy'] = (df['co2'] - co2_baseline) / co2_per_person
    df['estimated_occupancy'] = df['estimated_occupancy'].clip(lower=0)
    return df


# --- Thermal Comfort Analysis ---
def analyze_thermal_comfort(df, ideal_temp_range=IDEAL_TEMP_RANGE):
    too_cold = df[df['temperature'] < ideal_temp_range[0]]
    too_hot = df[df['temperature'] > ideal_temp_range[1]]
    insights = []
    if not too_cold.empty:
        insights.append(f"Temperature fell below the ideal range ({ideal_temp_range[0]}-{ideal_temp_range[1]}°C) "
                       f"at: {', '.join(too_cold.index.strftime('%Y-%m-%d %H:%M:%S').tolist())}")
    if not too_hot.empty:
        insights.append(f"Temperature exceeded the ideal range ({ideal_temp_range[0]}-{ideal_temp_range[1]}°C) "
                       f"at: {', '.join(too_hot.index.strftime('%Y-%m-%d %H:%M:%S').tolist())}")
    if too_cold.empty and too_hot.empty:
        insights.append("Temperature remained within the ideal comfort range.")
    return insights


# --- Noise Level Analysis ---
def analyze_noise_levels(df, noise_threshold=NOISE_THRESHOLD):
    insights = []
    high_noise_periods = df[df['noise_level'] > noise_threshold]
    if not high_noise_periods.empty:
        insights.append(f"Periods with high noise levels (above {noise_threshold} dB) were detected.")
    else:
        insights.append("Noise levels remained below the specified threshold.")
    return insights


# --- Energy Consumption Estimation ---
def calculate_energy_consumption(df, temperature_setpoint=22):
    temperature_difference = abs(df['temperature'] - temperature_setpoint)
    energy_consumption = temperature_difference.sum() * 0.1
    return energy_consumption


# --- Recommendations ---
def recommend_actions(df, insights):
    recommendations = []

    # CO2 Recommendations
    if any("high CO2" in insight.lower() for insight in insights):
        recommendations.append("Consider increasing ventilation to reduce CO2 levels.")
        if df['outside_temperature'].mean() < df['temperature'].mean() and \
                df['cloud_coverage'].mean() < 50:
            recommendations.append(
                "Open windows for natural ventilation as the outside temperature is cooler and it's relatively clear.")
        else:
            recommendations.append(
                "Consider using mechanical ventilation if opening windows is not feasible or effective.")
    if any("positive correlation" in insight.lower() for insight in insights) and \
            any("outside temperature" in insight.lower() for insight in insights):
        recommendations.append("Investigate if outdoor air quality is impacting indoor CO2 levels.")

    # Temperature Recommendations
    if df['temperature'].mean() > 25:
        recommendations.append("Consider lowering the thermostat to reduce energy consumption.")
        recommendations.append("Use ceiling fans to circulate air and create a cooling effect.")
    elif df['temperature'].mean() < 18:
        recommendations.append("Check heating system or consider using additional heating sources.")
        recommendations.append("Use space heaters in specific areas instead of heating the entire space.")

    # Humidity Recommendations
    if df['humidity'].mean() > 60:
        recommendations.append("Use a dehumidifier to prevent mold and improve air quality.")
        recommendations.append("Increase ventilation to allow moisture to escape.")
    elif df['humidity'].mean() < 30:
        recommendations.append("Use a humidifier to add moisture to the air and prevent dryness.")
        recommendations.append("Avoid using excessive heat sources, as they can dry out the air.")

    # Light Recommendations
    if df['light_intensity'].mean() < 100 and 8 <= datetime.datetime.now().hour <= 17:
        recommendations.append("Consider using additional lighting to improve visibility and comfort.")
        recommendations.append("Utilize natural light by opening curtains or blinds during daylight hours.")

    # Occupancy Recommendations
    if any("occupancy" in insight.lower() for insight in insights):
        if df['estimated_occupancy'].mean() > 5:
            recommendations.append("Consider increasing ventilation rates to accommodate higher occupancy.")
            recommendations.append("Implement occupancy-based lighting and HVAC controls.")
        else:
            recommendations.append("Adjust ventilation based on occupancy levels to optimize energy use.")

    # Thermal Comfort Recommendations
    if any("temperature" in insight.lower() for insight in insights):
        recommendations.append("Adjust thermostat settings or use fans for better temperature control.")
        recommendations.append("Consider using smart thermostats that learn your preferences and optimize energy use.")

    # VOC Recommendations
    if any("high voc" in insight.lower() for insight in insights):
        recommendations.append("Investigate potential sources of VOC emissions and take steps to mitigate them.")
        recommendations.append("Consider using air purifiers with VOC filters to improve air quality.")
        recommendations.append("Use low-VOC paints, sealants, and cleaning products.")
        recommendations.append("Ventilate the space thoroughly after using products that release VOCs.")

    # Noise Recommendations
    if any("high noise" in insight.lower() for insight in insights):
        recommendations.append("Identify the source of noise and take steps to reduce it.")
        recommendations.append("Use sound-absorbing materials on walls and ceilings.")
        recommendations.append("Install noise-canceling windows or doors.")
        recommendations.append("Encourage quieter activities during noise-sensitive periods.")

    # General Recommendations
    if df['deforestation_rate'].mean() > 0.05:
        recommendations.append("Consider supporting sustainable forestry practices.")
    if df['air_quality_index'].mean() > 100:
        recommendations.append("Promote carpooling, cycling, and public transport to reduce emissions.")

    return recommendations


# --- Visualization ---
def visualize_data(df):
    # Create a figure with subplots for each parameter
    fig, axes = plt.subplots(7, 1, figsize=(12, 12))  # Add a subplot for VOC
    fig.suptitle("Environmental Monitoring Data")

    # Plot Temperature
    axes[0].plot(df.index, df['temperature'], label="Temperature (°C)")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].grid(True)
    axes[0].legend()

    # Plot Humidity
    axes[1].plot(df.index, df['humidity'], label="Humidity (%)")
    axes[1].set_ylabel("Humidity (%)")
    axes[1].grid(True)
    axes[1].legend()

    # Plot CO2
    axes[2].plot(df.index, df['co2'], label="CO2 (ppm)")
    axes[2].set_ylabel("CO2 (ppm)")
    axes[2].grid(True)
    axes[2].legend()

    # Plot Light Intensity
    axes[3].plot(df.index, df['light_intensity'], label="Light Intensity")
    axes[3].set_ylabel("Light Intensity")
    axes[3].grid(True)
    axes[3].legend()

    # Plot Noise Level
    axes[4].plot(df.index, df['noise_level'], label="Noise Level (dB)")
    axes[4].set_ylabel("Noise Level (dB)")
    axes[4].grid(True)
    axes[4].legend()

    # Plot Estimated Occupancy
    axes[5].plot(df.index, df['estimated_occupancy'], label="Estimated Occupancy")
    axes[5].set_ylabel("Estimated Occupancy")
    axes[5].grid(True)
    axes[5].legend()

    # Plot VOC
    axes[6].plot(df.index, df['voc'], label="VOC")
    axes[6].set_ylabel("VOC")
    axes[6].grid(True)
    axes[6].legend()

    plt.tight_layout()
    plt.show()

    # Air Quality Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df.index, df['air_quality_index'], label="Air Quality Index (AQI)")
    plt.ylabel("AQI")
    plt.xlabel("Time")
    plt.title("Air Quality Index Trends")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Deforestation Rate (Bar Plot because it's a single value over time)
    plt.figure(figsize=(8, 6))
    plt.bar(df.index, df['deforestation_rate'], label="Deforestation Rate")
    plt.ylabel("Deforestation Rate")
    plt.xlabel("Time")
    plt.title("Deforestation Rate")
    plt.grid(True)
    plt.legend()
    plt.show()

    # OpenAQ Air Quality Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df.index, df['openaq_pm25'], label="PM2.5 (µg/m3)")
    plt.ylabel("PM2.5 (µg/m3)")
    plt.xlabel("Time")
    plt.title("OpenAQ PM2.5 Trends")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Precipitation Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df.index, df['precipitation'], label="Precipitation")
    plt.ylabel("Precipitation")
    plt.xlabel("Time")
    plt.title("Precipitation Trends")
    plt.grid(True)
    plt.legend()
    plt.show()


def get_deforestation_facts():
    """Provides general facts and figures about deforestation."""
    facts = [
        "About 10 million hectares of forest are lost each year.",
        "Deforestation contributes to around 15% of global greenhouse gas emissions.",
        "Protecting forests is crucial for biodiversity, climate regulation, and water resources."
    ]
    return np.random.choice(facts)


# --- Reporting ---
def generate_report(df, insights, recommendations, noise_insights, energy_consumption, aqi, filename_suffix=''):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Environmental Monitoring Report", ln=1, align="C")

    cell_height = 6
    
    # Air Quality Section
    pdf.ln(10)
    pdf.cell(200, 10, txt="Air Quality", ln=1, align="L")
    pdf.set_font("Arial", size=10)
    pdf.cell(200, cell_height, txt=f"Current Air Quality Index (AQI): {aqi}", ln=1, align="L")
    pdf.multi_cell(0, cell_height, txt="The Air Quality Index (AQI) is a measure of how polluted the air currently is. "
                              "Higher AQI values indicate greater pollution levels. ")

    # Environmental Impact Section
    pdf.ln(10)
    pdf.cell(200, 10, txt="Environmental Impact", ln=1, align="L")
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 5, txt="Indoor environmental quality is linked to global challenges. "
                              "High CO2 can mean more energy use and greenhouse gas emissions.\n\n"
                              f"Deforestation Fact: {get_deforestation_facts()}\n\n"
                              "Air quality is impacted by factors like pollution and climate change. "
                              "Learn more and take action: https://www.epa.gov/climatechange")

    pdf.ln(10)
    pdf.cell(200, 10, txt="Data Summary", ln=1, align="L")
    pdf.set_font("Arial", size=8)
    
    # Create a table header
    pdf.cell(40, cell_height, "Timestamp", 1) 
    pdf.cell(15, cell_height, "Temp (°C)", 1)
    pdf.cell(15, cell_height, "Humidity (%)", 1)
    pdf.cell(15, cell_height, "CO2 (ppm)", 1)
    pdf.cell(20, cell_height, "Light Intensity", 1)
    pdf.cell(15, cell_height, "Noise (dB)", 1)
    pdf.cell(20, cell_height, "Occupancy (Est.)", 1)
    pdf.cell(10, cell_height, "VOC", 1)  # Add VOC column
    pdf.ln()
    
    for index, row in df.iterrows():
        pdf.cell(40, cell_height, str(index), 1) 
        pdf.cell(15, cell_height, f"{row['temperature']:.2f}", 1)
        pdf.cell(15, cell_height, f"{row['humidity']:.2f}", 1)
        pdf.cell(15, cell_height, f"{row['co2']:.2f}", 1)
        pdf.cell(20, cell_height, f"{row['light_intensity']:.2f}", 1)
        pdf.cell(15, cell_height, f"{row['noise_level']:.2f}", 1)
        pdf.cell(20, cell_height, f"{row['estimated_occupancy']:.2f}", 1)
        pdf.cell(10, cell_height, f"{row['voc']:.2f}", 1)
        pdf.ln()

    # Satellite Monitoring Section
    pdf.ln(10)
    pdf.cell(200, 10, txt="Satellite Monitoring", ln=1, align="L")
    pdf.set_font("Arial", size=10)
    deforestation_rate_formatted = f"{df['deforestation_rate'].iloc[-1]:.4f}" if not df[
        'deforestation_rate'].empty else "N/A"
    pdf.multi_cell(0, 5, txt="Satellite images are being monitored for deforestation activities. "
                              f"The data shows a deforestation rate of: {deforestation_rate_formatted}")

    # OpenAQ Air Quality Data 
    pdf.ln(10)
    pdf.cell(200, 10, txt="OpenAQ Air Quality (PM2.5)", ln=1, align="L")
    pdf.set_font("Arial", size=10)
    pm25_value = df['openaq_pm25'].iloc[-1] if not df['openaq_pm25'].empty else "N/A"
    pdf.cell(200, 5, txt=f"Latest PM2.5 Value: {pm25_value} µg/m3", ln=1, align="L") 

    # Insights Section
    pdf.ln(10)
    pdf.cell(200, 10, txt="Insights", ln=1, align="L")
    pdf.set_font("Arial", size=10)
    for insight in insights:
        pdf.multi_cell(0, 5, txt=f"- {insight}")

    # Thermal Comfort Insights Section
    pdf.ln(10)
    pdf.cell(200, 10, txt="Thermal Comfort Insights", ln=1, align="L")
    pdf.set_font("Arial", size=10)
    for insight in noise_insights:
        pdf.multi_cell(0, 5, txt=f"- {insight}")

    # Noise Level Insights Section
    pdf.ln(10)
    pdf.cell(200, 10, txt="Noise Level Insights", ln=1, align="L")
    pdf.set_font("Arial", size=10)
    for insight in noise_insights:
        pdf.multi_cell(0, 5, txt=f"- {insight}")

    # Recommendations Section
    pdf.ln(10)
    pdf.cell(200, 10, txt="Recommendations", ln=1, align="L")
    pdf.set_font("Arial", size=10)
    for recommendation in recommendations:
        pdf.multi_cell(0, 5, txt=f"- {recommendation}")

    # Energy Consumption Section
    pdf.ln(10)
    pdf.cell(200, 10, txt="Energy Consumption", ln=1, align="L")
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 5, txt=f"Estimated Energy Consumption: {energy_consumption:.2f} units", ln=1, align="L")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(REPORT_FOLDER, f"env_report_{timestamp}{filename_suffix}.pdf")
    pdf.output(filename, 'F')
    print(f"Report generated: {filename}")
    
if __name__ == "__main__":
    model = None
    roi_coordinates = [72.5714, 23.0225]

    try:
        ee.Authenticate()
        ee.Initialize(project='ecovision-430807')
        print("Earth Engine authentication successful!")
    except ee.EEException as e:
        print(f"Error connecting to Earth Engine: {e}")

    # --- Generate the deforestation map outside the loop
    ndvi_images = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterBounds(ee.Geometry.Point(roi_coordinates)) \
        .filterDate('2023-01-01', '2023-12-31') 

    deforestation_rate = analyze_deforestation(ndvi_images, ee.Geometry.Point(roi_coordinates))

    all_data = pd.DataFrame(
        columns=['timestamp', 'temperature', 'humidity', 'co2', 'light_intensity', 'noise_level', 'voc',
                 'outside_temperature', 'wind_speed', 'air_quality_index', 'deforestation_rate',
                 'relative_humidity', 'estimated_occupancy', 'hour', 'openaq_pm25', 'precipitation'])  

    while True:
        try:
            data = collect_data()
            processed_data = process_data(data)

            integrated_data = integrate_weather_data(processed_data.copy(), location="Ahmedabad")
            integrated_data = integrate_air_quality_data(integrated_data.copy(), zip_code="90001")

            # Integrate data for each timestamp
            for index, row in processed_data.iterrows():
                # Get data from OpenAQ API
                integrated_data = integrate_openaq_data(integrated_data.copy(), location="Ahmedabad")
                
                # Add deforestation rate from the initial calculation
                integrated_data['deforestation_rate'] = deforestation_rate 
    
            if model is None:
                model = develop_model(integrated_data)
            else:
                model = train_model(model, integrated_data)

            validate_model(model, integrated_data)
            predictions = make_predictions(model, integrated_data)

            monitor_data(integrated_data)
            integrated_data = estimate_occupancy(integrated_data)
            insights = generate_insights(integrated_data, model)
            thermal_comfort_insights = analyze_thermal_comfort(integrated_data)
            noise_insights = analyze_noise_levels(integrated_data)
            energy_consumption = calculate_energy_consumption(integrated_data)

            recommendations = recommend_actions(
                integrated_data, insights + thermal_comfort_insights + noise_insights
            )

            print("\n----- Environmental Insights -----")
            for insight in insights:
                print(f"* {insight}")

            print("\n----- Thermal Comfort Insights -----")
            for insight in thermal_comfort_insights:
                print(f"* {insight}")

            print("\n----- Noise Level Insights -----")
            for insight in noise_insights:
                print(f"* {insight}")

            print(f"\nEstimated Energy Consumption: {energy_consumption:.2f} units")

            print("\n----- Recommendations -----")
            if recommendations:
                for recommendation in recommendations:
                    print(f"* {recommendation}")
            else:
                print("* No specific recommendations at this time.")

            visualize_data(integrated_data)
            latest_aqi = integrated_data['air_quality_index'].iloc[-1] if not integrated_data[
                'air_quality_index'].empty else "N/A"
            generate_report(
                integrated_data, insights, recommendations, noise_insights, energy_consumption, latest_aqi
            )

            all_data = pd.concat([all_data, integrated_data])

            all_data.to_csv(DATA_CSV)

        except Exception as e:
            print(f"An error occurred: {e}")

        time.sleep(10)