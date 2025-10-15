# smart_home_app_pyqt6.py
import sys
import json
import os
import requests
import datetime
import time
import pandas as pd
import joblib
import numpy as np
import paho.mqtt.client as mqtt
from sklearn.linear_model import LogisticRegression
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QMessageBox, QGroupBox,
    QSpacerItem, QSizePolicy, QLineEdit, QProgressBar
)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt

# --- IMPORTANT: ML Model & Data Configuration ---
MODEL_FILE = 'lightingmodel.pkl'
CALIBRATED_MODEL_FILE = 'calibrated_lightingmodel.pkl'
DATASET_FILE = 'dataset.xlsx'
CALIBRATION_STATUS_FILE = 'calibration_status.json'
NORMALIZATION_FACTOR = 32768.0
CALIBRATION_SAMPLES_REQUIRED = 50
WEATHER_API_KEY = "0b3c2a533b511a7b352a58fce05c83e1"  # Replace with your OpenWeatherMap API key
# Latitude and Longitude for Kolkata
LAT = 22.5726
LON = 88.3639
MQTT_TOPIC = "sensor/adc/data"
IP_FILE = "mqtt_ips.txt"

# --- MQTT Client Thread ---
class MQTTClientThread(QThread):
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
            self.client.connect(self.broker_ip, 1883, 60)
            self.client.loop_forever()
        except Exception as e:
            self.connection_status_signal.emit(f"Connection failed: {e}")

    def on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback for when the client connects to the broker."""
        if rc == 0:
            client.subscribe(MQTT_TOPIC)
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

# --- Calibration Thread for non-blocking data collection ---
class CalibrationThread(QThread):
    """A separate thread to handle the calibration data collection."""
    calibration_complete = pyqtSignal(list)
    progress_updated = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_points = []
        self.running = True

    def run(self):
        """Waits for data to be added until the required number is reached."""
        print(f"Calibration thread started. Collecting {CALIBRATION_SAMPLES_REQUIRED} samples.")
        while self.running and len(self.data_points) < CALIBRATION_SAMPLES_REQUIRED:
            time.sleep(0.5)  # Wait for a new data point to arrive from the Pi

        # When enough data is collected, emit the signal
        if len(self.data_points) >= CALIBRATION_SAMPLES_REQUIRED:
            self.calibration_complete.emit(self.data_points)
        self.running = False

    def add_data_point(self, data):
        """Adds a new data point to the list and updates the progress bar."""
        if self.running and len(self.data_points) < CALIBRATION_SAMPLES_REQUIRED:
            self.data_points.append(data)
            self.progress_updated.emit(len(self.data_points))
            print(f"Collected data point {len(self.data_points)}/{CALIBRATION_SAMPLES_REQUIRED}...")

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

class SmartHomeApp(QMainWindow):
    """
    A PyQt6 application for a two-node smart home system.
    It now includes an ML model for light prediction, a calibration workflow,
    and displays PIR motion status.
    """
    def __init__(self):
        """Initializes the main application window and its components."""
        super().__init__()

        self.setWindowTitle("Smart Home Monitor & Calibrator")
        self.setGeometry(100, 100, 700, 500)

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # --- System State Variables ---
        self.model = None
        self.dataset = pd.DataFrame(columns=['adc_value', 'day', 'lights_on'])
        self.is_calibrating = False
        self.calibration_phase = None
        self.calibration_thread = None
        self.mqtt_thread = None

        # Calibration state persistence
        self.calibration_status = self.load_calibration_status()
        self.day_night_status = None

        # --- UI Components ---
        # Main Title
        title_label = QLabel("Smart Home System")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        main_layout.addWidget(title_label)

        # Connection Group
        conn_group = QGroupBox("Connection")
        conn_layout = QHBoxLayout()
        conn_group.setLayout(conn_layout)
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("Enter MQTT Broker IP (e.g., 192.168.0.205)")
        self.connect_button = QPushButton("Connect to Pi")
        self.connection_status_label = QLabel("Status: Initializing...")
        conn_layout.addWidget(QLabel("Pi IP Address:"))
        conn_layout.addWidget(self.ip_input)
        conn_layout.addWidget(self.connect_button)
        conn_layout.addWidget(self.connection_status_label)
        main_layout.addWidget(conn_group)

        # System Status Group
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout()
        status_group.setLayout(status_layout)

        # Individual status labels
        self.adc_label = QLabel("LDR ADC Value: ---")
        self.adc_label.setStyleSheet("font-size: 16px;")
        self.pir_label = QLabel("PIR Status: ---")
        self.pir_label.setStyleSheet("font-size: 16px;")
        self.day_night_label = QLabel("Day/Night Status: Fetching...")
        self.day_night_label.setStyleSheet("font-size: 16px;")

        # Prediction label (styled to stand out)
        self.prediction_label = QLabel("Waiting for data...")
        self.prediction_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50;") # Green text

        status_layout.addWidget(self.adc_label)
        status_layout.addWidget(self.pir_label)
        status_layout.addWidget(self.day_night_label)
        status_layout.addWidget(self.prediction_label)
        main_layout.addWidget(status_group)

        # Calibration Group
        calibration_group = QGroupBox("Automated Calibration")
        calibration_layout = QVBoxLayout()
        calibration_group.setLayout(calibration_layout)
        self.calibration_status_label = QLabel("Current Status: Initializing...")
        self.calibration_status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.lights_off_button = QPushButton("1. Calibrate: Lights OFF")
        self.lights_on_button = QPushButton("2. Calibrate: Lights ON")
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(CALIBRATION_SAMPLES_REQUIRED)
        self.progress_bar.setVisible(False) # Hide until calibration starts

        calibration_layout.addWidget(self.calibration_status_label)
        calibration_layout.addWidget(self.lights_off_button)
        calibration_layout.addWidget(self.lights_on_button)
        calibration_layout.addWidget(self.progress_bar)

        self.connect_button.clicked.connect(self.connect_to_pi)
        self.lights_off_button.clicked.connect(lambda: self.start_calibration('lights_off'))
        self.lights_on_button.clicked.connect(lambda: self.start_calibration('lights_on'))

        main_layout.addWidget(calibration_group)
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # --- Timers and Event Loops ---
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(30000)
        self.update_timer.timeout.connect(self.update_day_night_status)
        self.update_timer.start()

        # --- Initial Setup ---
        self.update_day_night_status()
        self.load_model_and_dataset()
        self.update_calibration_ui()
        self.check_calibration_alert()
        self.attempt_auto_connect()

    def attempt_auto_connect(self):
        """Reads IPs from a file and attempts to connect to them."""
        self.connection_status_label.setText("Attempting to auto-connect...")
        if os.path.exists(IP_FILE):
            with open(IP_FILE, 'r') as f:
                ips = [line.strip() for line in f if line.strip()]

            for ip in reversed(ips):  # Try the most recent IP first
                print(f"Trying saved IP: {ip}...")
                self.ip_input.setText(ip)
                self.connect_to_pi()
                # Give it a short moment to connect
                time.sleep(1)
                if self.mqtt_thread and self.mqtt_thread.client.is_connected():
                    print(f"Successfully connected to saved IP: {ip}")
                    self.connection_status_label.setText(f"Connected to saved IP: {ip}")
                    return
                else:
                    print(f"Failed to connect to {ip}")
                    if self.mqtt_thread:
                        self.mqtt_thread.stop()

        self.connection_status_label.setText("No saved IPs connected. Please enter an IP.")

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

    def connect_to_pi(self):
        """Starts the MQTT client with the user-provided IP."""
        ip_address = self.ip_input.text()
        if not ip_address:
            QMessageBox.warning(self, "Invalid IP", "Please enter a valid IP address.")
            return

        if self.mqtt_thread and self.mqtt_thread.isRunning():
            self.mqtt_thread.stop()
            self.mqtt_thread = None

        self.mqtt_thread = MQTTClientThread(parent=self)
        self.mqtt_thread.data_received.connect(self.on_data_received)
        self.mqtt_thread.connection_status_signal.connect(self.update_connection_status)

        self.mqtt_thread.connect_to_broker(ip_address)
        self.save_ip_to_file(ip_address)

    def update_connection_status(self, status):
        """Updates the UI label with the connection status."""
        self.connection_status_label.setText(f"Status: {status}")

    def get_day_night_status(self):
        """Determines day/night status using OpenWeatherMap API."""
        if not WEATHER_API_KEY or WEATHER_API_KEY == "YOUR_API_KEY":
            print("Warning: API key not set. Assuming daytime.")
            return 0
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={WEATHER_API_KEY}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            sunrise = datetime.datetime.fromtimestamp(data['sys']['sunrise'])
            sunset = datetime.datetime.fromtimestamp(data['sys']['sunset'])
            current_time = datetime.datetime.now()
            return 0 if sunrise < current_time < sunset else 1
        except Exception as e:
            print(f"Error fetching day/night status: {e}")
            return 0 # Default to daytime on error

    def update_day_night_status(self):
        """Updates the day/night status and UI label."""
        old_status = self.day_night_status
        self.day_night_status = self.get_day_night_status()
        status_text = 'Day' if self.day_night_status == 0 else 'Night'
        self.day_night_label.setText(f"Day/Night Status: {status_text}")
        print(f"Updated day/night status to: {status_text}")

        if old_status is not None and old_status != self.day_night_status:
            self.check_calibration_alert()
            self.update_calibration_ui()

    def load_calibration_status(self):
        """Loads calibration status from file."""
        if os.path.exists(CALIBRATION_STATUS_FILE):
            try:
                with open(CALIBRATION_STATUS_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading calibration status: {e}")
        return {"night_calibrated": False, "day_calibrated": False}

    def save_calibration_status(self):
        """Saves calibration status to file."""
        try:
            with open(CALIBRATION_STATUS_FILE, 'w') as f:
                json.dump(self.calibration_status, f)
            print("Calibration status saved.")
        except Exception as e:
            print(f"Error saving calibration status: {e}")

    def check_calibration_alert(self):
        """
        Shows an alert if the time of day changes and calibration for that period
        has not been completed.
        """
        if self.day_night_status == 0 and self.calibration_status["night_calibrated"] and not self.calibration_status["day_calibrated"]:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Calibration Required")
            msg_box.setText("It is now daytime. Please perform the daytime calibration to complete the model training.")
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.exec()
        elif self.day_night_status == 1 and self.calibration_status["day_calibrated"] and not self.calibration_status["night_calibrated"]:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Calibration Required")
            msg_box.setText("It is now nighttime. Please perform the nighttime calibration to complete the model training.")
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.exec()

    def load_model_and_dataset(self):
        """
        Loads the calibrated model if it exists, otherwise falls back to the original model.
        Also loads the dataset.
        """
        if os.path.exists(CALIBRATED_MODEL_FILE):
            try:
                self.model = joblib.load(CALIBRATED_MODEL_FILE)
                print("Calibrated ML model loaded successfully.")
            except Exception as e:
                print(f"Error loading calibrated model: {e}. Falling back to original.")
                self.model = self.load_original_model()
        else:
            self.model = self.load_original_model()

        if os.path.exists(DATASET_FILE):
            try:
                self.dataset = pd.read_excel(DATASET_FILE)
                print("Dataset loaded successfully.")
            except Exception as e:
                print(f"Error loading dataset: {e}")
                self.dataset = pd.DataFrame(columns=['adc_value', 'day', 'lights_on'])
        else:
            self.dataset = pd.DataFrame(columns=['adc_value', 'day', 'lights_on'])

    def load_original_model(self):
        """Helper function to load the original pre-trained model."""
        if os.path.exists(MODEL_FILE):
            try:
                print("Original ML model loaded successfully.")
                return joblib.load(MODEL_FILE)
            except Exception as e:
                print(f"Error loading original model: {e}")
        return None

    def on_data_received(self, data):
        """
        Slot to handle the data emitted by the MQTT client thread.
        This function now also feeds data to the calibration thread and updates
        the LDR and PIR status labels.
        """
        # Extract data from the payload
        adc_value = data.get('adc_value')
        pir_status = data.get('pir_status')

        # Update the UI labels for LDR and PIR status
        if adc_value is not None:
            self.adc_label.setText(f"LDR ADC Value: {adc_value}")
        if pir_status is not None:
            status_text = "Motion Detected" if pir_status == 1 else "No Motion"
            self.pir_label.setText(f"PIR Status: {status_text}")

        if self.is_calibrating and self.calibration_thread:
            self.calibration_thread.add_data_point({
                'adc_value': adc_value,
                'day': self.day_night_status
            })
        else:
            # Use the ML model for prediction
            if adc_value is not None:
                self.predict_light_status(adc_value)

    def predict_light_status(self, adc_value):
        """Uses the loaded ML model to predict light status."""
        if self.model:
            try:
                normalized_adc = adc_value / NORMALIZATION_FACTOR
                input_data = np.array([[normalized_adc, self.day_night_status]])
                prediction = self.model.predict(input_data)[0]
                prediction_text = "Lights ON" if prediction == 1 else "Lights OFF"
                self.prediction_label.setText(prediction_text)
                print(f"Prediction for ADC:{adc_value}, Day/Night:{self.day_night_status} -> {prediction_text}")
            except Exception as e:
                print(f"Error making prediction: {e}")
                self.prediction_label.setText("Error")
        else:
            self.prediction_label.setText("Model not loaded. Please calibrate.")

    def start_calibration(self, phase):
        """Initiates a new calibration phase."""
        if self.mqtt_thread is None or not self.mqtt_thread.client.is_connected():
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Connection Required")
            msg_box.setText("Please connect to the Raspberry Pi before starting calibration.")
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.exec()
            return

        if self.day_night_status == 0 and self.calibration_status['day_calibrated']:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Daytime Already Calibrated")
            msg_box.setText("Daytime calibration has already been completed.")
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.exec()
            return
        elif self.day_night_status == 1 and self.calibration_status['night_calibrated']:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Nighttime Already Calibrated")
            msg_box.setText("Nighttime calibration has already been completed.")
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.exec()
            return

        if self.is_calibrating:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Calibration In Progress")
            msg_box.setText("Calibration is already running. Please wait for it to complete.")
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.exec()
            return

        self.is_calibrating = True
        self.calibration_phase = phase
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        msg_box = QMessageBox()
        msg_box.setWindowTitle("Start Calibration")

        if phase == 'lights_off':
            msg_box.setText("Please turn off all lights in the room, then press OK to start data collection.")
            self.calibration_status_label.setText(f"Status: Collecting 'Lights OFF' samples...")
        elif phase == 'lights_on':
            msg_box.setText("Please turn on all lights in the room, then press OK to start data collection.")
            self.calibration_status_label.setText(f"Status: Collecting 'Lights ON' samples...")

        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        if msg_box.exec() == QMessageBox.StandardButton.Ok:
            self.calibration_thread = CalibrationThread()
            self.calibration_thread.calibration_complete.connect(lambda data_points: self.end_calibration(data_points, phase))
            self.calibration_thread.progress_updated.connect(self.progress_bar.setValue)
            self.calibration_thread.start()
        else:
            self.is_calibrating = False
            self.calibration_phase = None
            self.progress_bar.setVisible(False)
            self.update_calibration_ui()

    def end_calibration(self, data_points, phase):
        """
        Called when a calibration phase is complete.
        This merges the new data, retrains the model, and saves everything.
        """
        self.is_calibrating = False
        self.progress_bar.setVisible(False)
        print(f"Calibration for phase '{phase}' complete. Collected {len(data_points)} samples.")

        lights_on_label = 1 if phase == 'lights_on' else 0
        new_data_df = pd.DataFrame([
            {'adc_value': d['adc_value'], 'day': d['day'], 'lights_on': lights_on_label}
            for d in data_points
        ])

        self.dataset = pd.concat([self.dataset, new_data_df], ignore_index=True)

        try:
            self.dataset.to_excel(DATASET_FILE, index=False)
            print(f"Updated dataset saved to '{DATASET_FILE}'.")
        except Exception as e:
            print(f"Error saving dataset: {e}")

        self.retrain_model()

        if self.day_night_status == 1:
            self.calibration_status['night_calibrated'] = True
        elif self.day_night_status == 0:
            self.calibration_status['day_calibrated'] = True
        self.save_calibration_status()
        self.update_calibration_ui()

    def retrain_model(self):
        """Retrains the ML model with the current dataset."""
        print("Retraining the ML model...")
        if self.dataset.empty:
            print("Dataset is empty, cannot retrain.")
            return

        X = self.dataset[['adc_value', 'day']].values / NORMALIZATION_FACTOR
        y = self.dataset['lights_on'].values

        try:
            self.model = LogisticRegression(solver='liblinear', random_state=42)
            self.model.fit(X, y)
            joblib.dump(self.model, CALIBRATED_MODEL_FILE)
            print(f"Model retrained and saved to '{CALIBRATED_MODEL_FILE}' successfully!")
        except Exception as e:
            print(f"Error during model retraining: {e}")
            self.model = None

    def update_calibration_ui(self):
        """Updates the UI based on the current calibration status."""
        status_text = "Current Status: "
        current_time_is_day = self.day_night_status == 0
        day_calibrated = self.calibration_status["day_calibrated"]
        night_calibrated = self.calibration_status["night_calibrated"]

        if day_calibrated and night_calibrated:
            status_text += "Fully calibrated (Day & Night)."
            self.lights_on_button.setEnabled(False)
            self.lights_off_button.setEnabled(False)
        elif current_time_is_day and not day_calibrated:
            status_text += "Initial daytime calibration required."
            self.lights_on_button.setEnabled(True)
            self.lights_off_button.setEnabled(True)
        elif not current_time_is_day and not night_calibrated:
            status_text += "Initial nighttime calibration required."
            self.lights_on_button.setEnabled(True)
            self.lights_off_button.setEnabled(True)
        elif day_calibrated and not night_calibrated:
            status_text += "Day calibration complete. Awaiting nighttime calibration."
            self.lights_on_button.setEnabled(False)
            self.lights_off_button.setEnabled(False)
        elif night_calibrated and not day_calibrated:
            status_text += "Night calibration complete. Awaiting daytime calibration."
            self.lights_on_button.setEnabled(False)
            self.lights_off_button.setEnabled(False)

        self.calibration_status_label.setText(status_text)

    def closeEvent(self, event):
        """Clean up the threads when the application closes."""
        if self.calibration_thread:
            self.calibration_thread.stop()
        if self.mqtt_thread:
            self.mqtt_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SmartHomeApp()
    window.show()
    sys.exit(app.exec())
