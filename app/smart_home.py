import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QMessageBox
)
from PyQt6.QtCore import QTimer, QDateTime, Qt
from datetime import datetime

class SmartHomeApp(QMainWindow):
    """
    A PyQt6 application for a two-node smart home system.

    It monitors simulated motion sensor nodes and provides an alert
    if there is no activity for a set period and it's nighttime.
    """

    def __init__(self):
        """Initializes the main application window and its components."""
        super().__init__()

        self.setWindowTitle("Smart Home Monitor")
        self.setGeometry(100, 100, 500, 300)

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # --- System State Variables ---
        self.nodes = {
            1: {"last_motion": datetime.now(), "is_motion": False},
            2: {"last_motion": datetime.now(), "is_motion": False},
        }
        # Day/night status is now determined by the system clock
        self.day_night_status = 0  
        self.inactivity_threshold_minutes = 5
        self.alerted_nodes = {1: False, 2: False}

        # --- UI Components ---
        self.status_labels = {}
        for i in range(1, 3):
            label_box = QHBoxLayout()
            node_label = QLabel(f"Node {i}: ")
            status_label = QLabel("Waiting for motion...")
            self.status_labels[i] = status_label
            label_box.addWidget(node_label)
            label_box.addWidget(status_label)
            main_layout.addLayout(label_box)

        # Day/Night Status Label (now without a toggle button)
        day_night_layout = QHBoxLayout()
        day_night_label = QLabel("Current Status:")
        self.day_night_status_label = QLabel("Daytime (0)")
        day_night_layout.addWidget(day_night_label)
        day_night_layout.addWidget(self.day_night_status_label)
        main_layout.addLayout(day_night_layout)

        # Motion Sensor Buttons (simulated input)
        button_layout = QHBoxLayout()
        self.node1_button = QPushButton("Motion in Node 1")
        self.node2_button = QPushButton("Motion in Node 2")
        self.node1_button.clicked.connect(lambda: self.on_motion_detected(1))
        self.node2_button.clicked.connect(lambda: self.on_motion_detected(2))
        button_layout.addWidget(self.node1_button)
        button_layout.addWidget(self.node2_button)
        main_layout.addLayout(button_layout)

        # --- Timers and Event Loops ---
        # QTimer for periodically updating the UI and checking for alerts
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(1000)  # Update UI every second
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start()

        self.check_timer = QTimer(self)
        # Check for alerts every 10 seconds (for demonstration)
        # You can increase this to 60000 for a one-minute check
        self.check_timer.setInterval(10000)
        self.check_timer.timeout.connect(self.check_for_alerts)
        self.check_timer.start()

        # Timer to automatically update the day/night status every minute
        self.day_night_timer = QTimer(self)
        self.day_night_timer.setInterval(60000) # Check every minute
        self.day_night_timer.timeout.connect(self.update_day_night_status)
        self.day_night_timer.start()

        # Initial check for day/night status on startup
        self.update_day_night_status()

    def update_day_night_status(self):
        """
        Automatically updates the day/night status based on the current system time.
        
        This method will eventually be replaced by the output of your ML model.
        For now, it uses a simple 6 AM to 6 PM rule.
        """
        current_hour = datetime.now().hour
        if 6 <= current_hour < 18:
            self.day_night_status = 0  # Daytime
            self.day_night_status_label.setText("Daytime (0)")
        else:
            self.day_night_status = 1  # Nighttime
            self.day_night_status_label.setText("Nighttime (1)")

    def on_motion_detected(self, node_id):
        """Updates the motion status and timestamp for a given node."""
        current_time = datetime.now()
        self.nodes[node_id]["last_motion"] = current_time
        self.nodes[node_id]["is_motion"] = True
        print(f"Motion detected in Node {node_id} at {current_time.strftime('%H:%M:%S')}")

    def update_ui(self):
        """Updates the status labels on the GUI."""
        for node_id, data in self.nodes.items():
            last_motion = data["last_motion"]
            time_diff = datetime.now() - last_motion
            minutes_inactive = time_diff.total_seconds() / 60
            
            # Change the state from "motion" to "no motion" after 5 seconds
            # This is to make the UI responsive
            if data["is_motion"] and minutes_inactive * 60 > 5:
                data["is_motion"] = False

            if data["is_motion"]:
                status_text = "Motion detected!"
            else:
                status_text = f"No motion. Last seen {minutes_inactive:.1f} mins ago."
            
            self.status_labels[node_id].setText(status_text)

    def check_for_alerts(self):
        """
        Checks if the alert conditions are met for any node.
        
        The alert condition is:
        1. It is nighttime (day_night_status == 1).
        2. A node has been inactive for more than the threshold.
        3. The alert has not already been shown for this node.
        """
        print("Checking for alerts...")
        
        if self.day_night_status == 1:
            for node_id, data in self.nodes.items():
                time_diff = datetime.now() - data["last_motion"]
                minutes_inactive = time_diff.total_seconds() / 60

                if minutes_inactive > self.inactivity_threshold_minutes and not self.alerted_nodes[node_id]:
                    # The condition is met, show the alert
                    alert_message = (
                        f"ALERT! No activity in Node {node_id} for "
                        f"over {self.inactivity_threshold_minutes} minutes. "
                        f"Please turn off all appliances in that area."
                    )
                    
                    # Use a non-blocking message box to show the alert
                    msg_box = QMessageBox()
                    msg_box.setWindowTitle("Smart Home Alert")
                    msg_box.setText(alert_message)
                    msg_box.setIcon(QMessageBox.Icon.Warning)
                    msg_box.exec()
                    
                    self.alerted_nodes[node_id] = True
                    print(alert_message)
        
        # Reset the alert flag if motion is detected again
        for node_id, data in self.nodes.items():
            time_diff = datetime.now() - data["last_motion"]
            minutes_inactive = time_diff.total_seconds() / 60
            if minutes_inactive < self.inactivity_threshold_minutes:
                self.alerted_nodes[node_id] = False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SmartHomeApp()
    window.show()
    sys.exit(app.exec())
