# smart_home_monitor.py
import sys
import json
import time
import requests
import joblib
import numpy as np
import paho.mqtt.client as mqtt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont
import os
from datetime import datetime

# --- IMPORTANT: MQTT Configuration ---
# This will be dynamically set, but provides a default
BROKER_IP = "" 
PORT = 1883
TOPIC = "sensor/adc/data"
IP_FILE = "mqtt_ips.txt"

# --- OpenWeatherMap API Configuration (Your API Key) ---
API_KEY = "0b3c2a533b511a7b352a58fce05c83e1"
LATITUDE = "22.5726"
LONGITUDE = "88.3639"

# --- Machine Learning Model Setup ---
try:
    ML_MODEL = joblib.load('lightingmodel.pkl')
    print("Machine learning model 'lightingmodel.pkl' loaded successfully.")
except FileNotFoundError:
    ML_MODEL = None
    print("Error: ML model file 'lightingmodel.pkl' not found. Predictions will be disabled.")


# --- MQTT Thread to Prevent GUI Freezing ---
class MqttThread(QThread):
    """
    A dedicated thread to handle MQTT communication.
    This prevents the GUI from freezing while waiting for messages.
    """
    data_received = pyqtSignal(dict)
    connection_status_signal = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.broker_ip = ""
        self.running = True

    def run(self):
        """Starts the MQTT client and connects to the broker."""
        if not self.broker_ip:
            self.connection_status_signal.emit("No IP address specified.")
            return

        self.connection_status_signal.emit(f"Connecting to {self.broker_ip}...")
        try:
            self.client.connect(self.broker_ip, PORT, 60)
            self.client.loop_forever()
        except Exception as e:
            self.connection_status_signal.emit(f"Connection failed: {e}")

    def on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback for when the client connects to the broker."""
        if rc == 0:
            client.subscribe(TOPIC)
            self.connection_status_signal.emit(f"Connected to {self.broker_ip}")
        else:
            self.connection_status_signal.emit(f"Connection failed, return code: {rc}")
            print(f"Failed to connect, return code: {rc}")

    def on_message(self, client, userdata, msg):
        """Callback for when a message is received."""
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            self.data_received.emit(payload)
        except json.JSONDecodeError:
            print("Error: Could not decode JSON payload.")
        except Exception as e:
            print(f"An error occurred in on_message: {e}")

    def connect_to_broker(self, ip_address):
        """Public method to start the connection process."""
        self.broker_ip = ip_address
        self.start()

    def stop(self):
        """Stops the MQTT client thread gracefully."""
        self.client.loop_stop()
        self.client.disconnect()
        self.wait()

# --- Main Application Window ---
class SmartHomeMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Home Monitor")
        self.setGeometry(100, 100, 400, 400)
        
        self.mqtt_thread = None
        self.connected_ip = ""

        self.init_ui()
        self.attempt_auto_connect()

    def init_ui(self):
        """Initializes the main UI components."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        title_font = QFont("Arial", 18, QFont.Weight.Bold)
        label_font = QFont("Arial", 14)
        status_font = QFont("Arial", 16, QFont.Weight.Bold)
        
        # IP Input Section
        ip_layout = QHBoxLayout()
        self.ip_input = QLineEdit()
        self.ip_input.setFont(label_font)
        self.ip_input.setPlaceholderText("Enter MQTT Broker IP (e.g., 192.168.0.205)")
        self.connect_button = QPushButton("Connect")
        self.connect_button.setFont(label_font)
        self.connect_button.clicked.connect(self.on_connect_button_clicked)
        ip_layout.addWidget(self.ip_input)
        ip_layout.addWidget(self.connect_button)
        main_layout.addLayout(ip_layout)
        
        self.connection_status_label = QLabel("Initializing...")
        self.connection_status_label.setFont(QFont("Arial", 12))
        self.connection_status_label.setStyleSheet("color: gray;")
        main_layout.addWidget(self.connection_status_label)
        
        main_layout.addSpacing(20)

        # LDR Section
        self.adc_label = QLabel("LDR ADC Value: ---")
        self.adc_label.setFont(label_font)
        main_layout.addWidget(self.adc_label)

        # Day/Night Section
        self.day_night_label = QLabel("Day/Night Status: ---")
        self.day_night_label.setFont(label_font)
        main_layout.addWidget(self.day_night_label)

        # PIR Section
        self.pir_label = QLabel("PIR Status: ---")
        self.pir_label.setFont(label_font)
        main_layout.addWidget(self.pir_label)

        # ML Prediction Section
        self.prediction_label = QLabel("ML Prediction: ---")
        self.prediction_label.setFont(status_font)
        self.prediction_label.setStyleSheet("color: #4CAF50;") # Green text
        main_layout.addWidget(self.prediction_label)
        
        main_layout.addStretch()

    def attempt_auto_connect(self):
        """Reads IPs from a file and attempts to connect to them."""
        self.connection_status_label.setText("Attempting to auto-connect...")
        if os.path.exists(IP_FILE):
            with open(IP_FILE, 'r') as f:
                ips = [line.strip() for line in f if line.strip()]
            
            for ip in ips:
                try:
                    self.connection_status_label.setText(f"Trying saved IP: {ip}...")
                    QApplication.processEvents() # Update GUI
                    
                    self.mqtt_thread = MqttThread()
                    self.mqtt_thread.data_received.connect(self.on_data_received)
                    self.mqtt_thread.connection_status_signal.connect(self.on_connection_status_update)
                    self.mqtt_thread.broker_ip = ip
                    self.mqtt_thread.client.connect(ip, PORT, 1) # Short timeout
                    
                    if self.mqtt_thread.client.is_connected():
                        self.connected_ip = ip
                        self.ip_input.setText(ip)
                        self.connection_status_label.setText(f"Connected to saved IP: {ip}")
                        print(f"Successfully connected to saved IP: {ip}")
                        self.mqtt_thread.client.loop_start()
                        return
                    else:
                        print(f"Failed to connect to {ip}")
                        self.mqtt_thread.stop()
                except Exception as e:
                    print(f"Exception during auto-connect to {ip}: {e}")
            
        self.connection_status_label.setText("No saved IPs connected. Please enter an IP.")

    def on_connect_button_clicked(self):
        """Handles the user's attempt to connect to a new IP."""
        new_ip = self.ip_input.text().strip()
        if not new_ip:
            QMessageBox.warning(self, "Invalid IP", "Please enter a valid IP address.")
            return

        # Stop existing connection if any
        if self.mqtt_thread and self.mqtt_thread.isRunning():
            self.mqtt_thread.stop()
            self.mqtt_thread.wait() # Ensure it's stopped before starting a new one

        self.connected_ip = new_ip
        self.mqtt_thread = MqttThread()
        self.mqtt_thread.data_received.connect(self.on_data_received)
        self.mqtt_thread.connection_status_signal.connect(self.on_connection_status_update)
        
        self.mqtt_thread.connect_to_broker(new_ip)
        
        # Save the new IP to the file if it's not already there
        self.save_ip_to_file(new_ip)

    def save_ip_to_file(self, ip):
        """Saves a new, unique IP address to the IP file."""
        ips = []
        if os.path.exists(IP_FILE):
            with open(IP_FILE, 'r') as f:
                ips = [line.strip() for line in f if line.strip()]
        
        if ip not in ips:
            with open(IP_FILE, 'a') as f:
                f.write(f"{ip}\n")
            print(f"Saved new IP to file: {ip}")
    
    def on_connection_status_update(self, status):
        """Updates the connection status label on the GUI."""
        self.connection_status_label.setText(status)

    def get_day_night_status(self):
        """
        Fetches sunrise/sunset from API and determines day or night.
        """
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={LATITUDE}&lon={LONGITUDE}&appid={API_KEY}"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            sunrise_utc = data['sys']['sunrise']
            sunset_utc = data['sys']['sunset']
            current_time_utc = int(time.time())
            
            if sunrise_utc < current_time_utc < sunset_utc:
                return 1 # Daytime
            else:
                return 0 # Nighttime
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return -1

    def make_prediction(self, adc_value, day_night_flag):
        """Uses the loaded ML model to make a prediction."""
        if ML_MODEL is None:
            return "Prediction Disabled"
        
        try:
            input_data = np.array([[adc_value, day_night_flag]])
            prediction = ML_MODEL.predict(input_data)[0]
            
            if prediction == 1:
                self.prediction_label.setStyleSheet("color: #03a9f4;") # Blue for ON
                return "Lights ON"
            else:
                self.prediction_label.setStyleSheet("color: #f44336;") # Red for OFF
                return "Lights OFF"
        except Exception as e:
            print(f"Error making prediction: {e}")
            return "Prediction Error"

    def on_data_received(self, data):
        """
        Slot to handle the data received from the MQTT thread.
        This updates the GUI labels with the new information.
        """
        try:
            # Extract data from the received payload
            adc_value = data.get("adc_value", "---")
            pir_status = data.get("pir_status", "---")
            
            # Update the ADC and PIR labels
            self.adc_label.setText(f"LDR ADC Value: {adc_value}")
            if pir_status == 1:
                self.pir_label.setText("PIR Status: Motion Detected")
            else:
                self.pir_label.setText("PIR Status: No Motion")
            
            # Get day/night status and make a prediction
            day_night_flag = self.get_day_night_status()
            day_text = "Daytime" if day_night_flag == 1 else "Nighttime"
            self.day_night_label.setText(f"Day/Night Status: {day_text}")
            
            prediction = self.make_prediction(adc_value, day_night_flag)
            self.prediction_label.setText(f"ML Prediction: {prediction}")
            
        except KeyError as e:
            print(f"Error: Missing key in payload: {e}")
        except Exception as e:
            print(f"Error updating UI: {e}")

    def closeEvent(self, event):
        """Clean up the MQTT thread when the application closes."""
        if self.mqtt_thread:
            self.mqtt_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SmartHomeMonitor()
    window.show()
    sys.exit(app.exec())
