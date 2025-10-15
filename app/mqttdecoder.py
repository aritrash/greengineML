# laptop_mqtt_subscriber.py
import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime
import requests
import joblib # Import joblib for loading the model
import numpy as np # Import numpy for data formatting

# --- IMPORTANT: Set the IP of the MQTT broker (your Pi's static IP) ---
BROKER_IP = "192.168.0.205"  # Use the correct IP from your Pi
PORT = 1883                  # Default MQTT port

# The topic to subscribe to.
TOPIC = "sensor/adc/data"

# --- OpenWeatherMap API Configuration ---
API_KEY = "0b3c2a533b511a7b352a58fce05c83e1"
LATITUDE = "22.5726"
LONGITUDE = "88.3639"

# --- Machine Learning Model Setup ---
# Load the trained ML model from the file
try:
    ml_model = joblib.load('lightingmodel.pkl')
    print("Machine learning model 'lightingmodel.pkl' loaded successfully.")
except FileNotFoundError:
    print("Error: ML model file 'lightingmodel.pkl' not found.")
    print("Please make sure the file is in the same directory.")
    ml_model = None # Set to None to handle errors gracefully

def get_day_night_status():
    """
    Fetches sunrise and sunset times from the OpenWeatherMap API and
    determines if it's currently day or night.
    Returns 1 for day, 0 for night.
    """
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={LATITUDE}&lon={LONGITUDE}&appid={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        sunrise_utc = data['sys']['sunrise']
        sunset_utc = data['sys']['sunset']
        current_time_utc = int(time.time())
        
        if sunrise_utc < current_time_utc < sunset_utc:
            return 1
        else:
            return 0
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return -1 # Return -1 to signal API failure

def on_connect(client, userdata, flags, rc, properties=None):
    """Callback function for when the client connects to the broker."""
    if rc == 0:
        print("Connected to MQTT Broker successfully!")
        client.subscribe(TOPIC)
        print(f"Subscribed to topic: {TOPIC}")
    else:
        print(f"Failed to connect, return code: {rc}")

def on_message(client, userdata, msg):
    """Callback function for when a message is received from the broker."""
    if ml_model is None:
        print("ML model is not loaded. Cannot make predictions.")
        return

    print(f"Received message on topic: {msg.topic}")
    
    try:
        # Parse the JSON payload from the Pi
        payload = json.loads(msg.payload.decode('utf-8'))
        
        # Extract both the ADC and PIR values
        adc_value = payload["adc_value"]
        pir_status = payload["pir_status"] # <-- New line to get PIR status
        
        # Get the day/night status from the API
        day_night_flag = get_day_night_status()

        # Check if the API call was successful
        if day_night_flag == -1:
            print("Could not get a reliable day/night status from API. Using a simple threshold.")
            day_night_flag = 1 if adc_value > 20000 else 0 # Example threshold

        # Prepare the input data for the model.
        # Ensure the data is in the correct format for your model.
        input_data = np.array([[adc_value, day_night_flag]])

        # Make a prediction using the loaded model
        prediction = ml_model.predict(input_data)

        # Print the results, including the PIR status
        print("-" * 20)
        print(f"LDR Raw Value: {adc_value}")
        print(f"PIR Status: {'Motion Detected' if pir_status == 1 else 'No Motion'}") # <-- Print PIR status
        print(f"Day/Night Flag: {day_night_flag}")
        print(f"ML Model Prediction: {prediction[0]}")
        
    except json.JSONDecodeError:
        print("Error: Could not decode JSON payload.")
    except KeyError as e:
        print(f"Error: Key '{e}' not found in payload. Check Pi's published JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Main script starts here ---
def main():
    """
    Main function to set up and run the MQTT client.
    """
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        print(f"Connecting to MQTT Broker at {BROKER_IP}:{PORT}...")
        client.connect(BROKER_IP, PORT, 60)
        client.loop_forever()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

