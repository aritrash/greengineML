# pi_mqtt_publisher.py
import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime

# Import GPIO and other necessary libraries
import RPi.GPIO as GPIO
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# --- IMPORTANT: Set the IP of the MQTT broker (your laptop's IP) ---
BROKER_IP = "192.168.0.205"  # Replace with your laptop's IP address
PORT = 1883  # Default MQTT port

# The topic to publish data to.
TOPIC = "sensor/adc/data"
# The topic to listen for control commands.
CONTROL_TOPIC = "control/relay/toggle"

# --- Hardware Setup ---
# I2C and ADS1115 for LDR
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    ads = ADS.ADS1115(i2c)
    ldr_channel = AnalogIn(ads, ADS.P0)
    print("ADS1115 sensor initialized successfully.")
except ValueError:
    print("Error: Could not initialize I2C bus or find the ADC.")
    print("       Please check your wiring and ensure I2C is enabled.")
    exit()

# PIR sensor setup
GPIO.setmode(GPIO.BCM)  # Use BCM GPIO numbering
pir_channel = 17       # GPIO17 (Pin 11) is a good choice for digital input
# The key change is here: add a pull-down resistor to prevent a 'floating' pin.
GPIO.setup(pir_channel, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) 
print("PIR sensor initialized successfully.")

# Relay setup
relay_pin = 23         # GPIO23 (Pin 16) - Adjust if needed
GPIO.setup(relay_pin, GPIO.OUT)
# Set the initial state to HIGH, which will turn OFF an active-low relay.
# This prevents the relay from clicking on at startup.
GPIO.output(relay_pin, GPIO.HIGH) 
print("Relay initialized successfully on GPIO 23.")

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc, properties=None):
    """Callback function for when the client connects to the broker."""
    if rc == 0:
        print("Connected to MQTT Broker!")
        # Subscribe to the control topic here
        client.subscribe(CONTROL_TOPIC)
        print(f"Subscribed to topic: {CONTROL_TOPIC}")
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    """
    Callback function for when a message is received from the broker.
    This function handles the control commands for the relay,
    now with inverted logic for an active-low relay.
    """
    if msg.topic == CONTROL_TOPIC:
        command = msg.payload.decode('utf-8').upper()
        print(f"Received command: {command}")
        if command == "ON":
            # For an active-low relay, a LOW signal turns it ON
            GPIO.output(relay_pin, GPIO.LOW)
            print("Relay turned ON.")
        elif command == "OFF":
            # For an active-low relay, a HIGH signal turns it OFF
            GPIO.output(relay_pin, GPIO.HIGH)
            print("Relay turned OFF.")
        else:
            print("Unknown command received.")

def send_data_to_broker():
    """
    Connects to the broker and continuously publishes sensor data.
    """
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message # Set the on_message callback

    try:
        # Connect to the MQTT broker
        client.connect(BROKER_IP, PORT, 60)

        # Start a loop to process network traffic and callbacks
        client.loop_start()

        print(f"Publishing data to topic: {TOPIC}")
        while True:
            # Read the raw 16-bit ADC value from the LDR channel
            adc_value = ldr_channel.value

            # Read the digital value from the PIR sensor
            pir_status = GPIO.input(pir_channel)
            
            # The Pi side will still use a simple 'day' flag for now,
            # as the laptop will overwrite this with the API data.
            day_flag = 0 
            
            # Create a dictionary for the data, including the PIR status
            data_to_send = {
                "adc_value": adc_value,
                "pir_status": pir_status, # <-- New key for PIR data
                "day": day_flag,
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            json_data = json.dumps(data_to_send)
            
            # Publish the JSON data
            client.publish(TOPIC, json_data, qos=1)
            print(f"Published: {json_data}")
            
            time.sleep(1)
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        GPIO.cleanup() # Clean up GPIO settings on exit
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    send_data_to_broker()
