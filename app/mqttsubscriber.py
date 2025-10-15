# laptop_mqtt_subscriber.py
import paho.mqtt.client as mqtt
import json
import pickle
import numpy as np

# --- IMPORTANT: Set the IP of the MQTT broker (your Pi's static IP) ---
BROKER_IP = "192.168.0.205" # This must be the Pi's IP address.
PORT = 1883              # Default MQTT port

# The topic to subscribe to. This must match the publisher's topic.
TOPIC = "sensor/adc/data"

# Normalization factor from your ML model training script
NORMALIZATION_FACTOR = 32768.0

# --- Load the ML model once when the script starts ---
try:
    with open('lightingmodel.pkl', 'rb') as file:
        model = pickle.load(file)
        print("ML Model loaded successfully!")
except FileNotFoundError:
    print("Error: lightingmodel.pkl not found. Please place the file in the same directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

def on_connect(client, userdata, flags, rc, properties=None):
    """The callback for when the client connects to the broker."""
    if rc == 0:
        print("Connected to MQTT Broker!")
        # Subscribe to the topic. The client will now receive messages on this topic.
        client.subscribe(TOPIC)
        print(f"Subscribed to topic: {TOPIC}")
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    """
    The callback for when a PUBLISH message is received from the broker.
    This function processes every incoming message.
    """
    print(f"\nMessage received on topic: {msg.topic}")
    try:
        # Decode the payload from bytes to a string and parse the JSON
        json_data = msg.payload.decode('utf-8')
        received_data = json.loads(json_data)

        adc_value = received_data.get("adc_value")
        day_flag = received_data.get("day")

        if adc_value is not None and day_flag is not None:
            # Normalize the ADC value, just like in your training script
            normalized_adc_value = adc_value / NORMALIZATION_FACTOR
            
            # The model expects a 2D array with two features: adc_value and day
            input_data = np.array([[normalized_adc_value, day_flag]])

            # Use the loaded model to make a prediction
            prediction_result = model.predict(input_data)
            
            # Based on the model's output (0 or 1), return a human-readable string.
            if prediction_result[0] == 0:
                prediction = "Lights Off"
            else:
                prediction = "Lights On"
            
            print(f"Received ADC value: {adc_value}")
            print(f"ML Model Prediction: {prediction}")
        else:
            print("Received JSON data is missing a required key ('adc_value' or 'day').")

    except json.JSONDecodeError:
        print(f"Could not decode JSON: {msg.payload}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

def start_subscriber():
    """
    Connects to the broker and starts listening for messages.
    """
    # Use CallbackAPIVersion.VERSION2 to avoid deprecation warnings
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER_IP, PORT, 60)
        # This will block the thread and keep the subscriber running
        client.loop_forever()
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    start_subscriber()
