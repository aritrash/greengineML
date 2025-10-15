# python_server_with_ml.py
import socket
import json
import pickle
import numpy as np

# --- IMPORTANT: Use the same IP and PORT as your Raspberry Pi client code ---
# The HOST is '0.0.0.0' to accept connections from any IP.
HOST = '0.0.0.0'
PORT = 5000

# We need to know the normalization factor used during training to apply it
# to the ADC value received from the Pi. This value should match the one
# used in your model training script.
NORMALIZATION_FACTOR = 32768.0

# --- Function to load and use your ML model ---
def predict_with_ml_model(adc_value):
    """
    Loads a pre-trained machine learning model and uses it to make a prediction.
    
    IMPORTANT: You must save your trained model as 'lightingmodel.pkl' and
    place it in the same directory as this script.
    
    Args:
        adc_value (int): The ADC reading received from the Raspberry Pi.
        
    Returns:
        str: A classification of the light environment ("Lights On" or "Lights Off").
        str: An error message if the model cannot be loaded.
    """
    try:
        # Load the saved model from the file
        # 'rb' means 'read binary' which is necessary for model files
        with open('lightingmodel.pkl', 'rb') as file:
            model = pickle.load(file)
            print("Model loaded successfully!")

        # The model was trained with normalized data, so we must normalize
        # the incoming ADC value before passing it to the model.
        normalized_adc_value = adc_value / NORMALIZATION_FACTOR
        
        # The model's input expects a 2D array with two features:
        # the normalized ADC value and the 'day' feature.
        # Since we're not receiving the 'day' feature from the Pi, we'll
        # assume it's a '1' to indicate daytime, as per your training data.
        input_data = np.array([[normalized_adc_value, 1]])

        # Use the loaded model to make a prediction
        prediction_result = model.predict(input_data)
        
        # Based on the model's output (0 or 1), return a human-readable string.
        # This part depends on what your model's prediction values are.
        # Based on your script, 0 is 'Lights Off' and 1 is 'Lights On'.
        if prediction_result[0] == 0:
            return "Lights Off"
        else:
            return "Lights On"

    except FileNotFoundError:
        return "Error: lightingmodel.pkl not found. Make sure the file is in the same directory."
    except Exception as e:
        return f"An error occurred while using the model: {e}"

def start_server():
    """
    Sets up a TCP server to listen for, receive, and process data from the client.
    """
    try:
        # Create a socket object using IPv4 and TCP
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Bind the socket to the specified host and port
            s.bind((HOST, PORT))
            print(f"Server started, listening on {HOST}:{PORT}")

            # Listen for incoming connections (allow one connection at a time)
            s.listen(1)
            
            # Wait for a client to connect
            conn, addr = s.accept()
            print(f"Connected by {addr}")

            # Handle the connection
            with conn:
                while True:
                    # Receive data from the client
                    data = conn.recv(1024)
                    if not data:
                        # If no data is received, the connection is closed
                        break

                    # Decode the received data and parse the JSON
                    json_data = data.decode('utf-8')
                    try:
                        received_data = json.loads(json_data)

                        # Extract the ADC value from the parsed JSON
                        adc_value = received_data.get("adc_value")
                        current_time = received_data.get("current_time")
                        
                        if adc_value is not None:
                            # Pass the ADC value to the ML model function
                            prediction = predict_with_ml_model(adc_value)
                            
                            # Print the raw data and the model's prediction
                            print(f"Received at {current_time}: ADC value is {adc_value}")
                            print(f"ML Model Prediction: {prediction}\n")
                        else:
                            print("Received JSON data is missing 'adc_value'.")

                    except json.JSONDecodeError:
                        print(f"Could not decode JSON: {json_data}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    start_server()
