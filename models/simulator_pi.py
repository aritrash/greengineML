import json
import random
import time
import socket

HOST = "127.0.0.1"  # Flask server IP
PORT = 5001         # Arbitrary TCP port

def simulate_room_data():
    """Simulate per-room motion, light, and connection status."""
    rooms = {}
    for i in range(1, 7):
        rooms[f"room{i}"] = {
            "motion": random.choice([True, False]),
            "light": random.choice([True, False]),
            "connected": True  # Can randomize False to simulate disconnection
        }
    return rooms

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        while True:
            data = simulate_room_data()
            packet = json.dumps(data)
            s.sendall(packet.encode("utf-8"))
            time.sleep(2)

if __name__ == "__main__":
    main()
