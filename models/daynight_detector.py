import requests
import datetime
import time

def get_day_night_status(api_key, city):
    """
    Determines if it is daytime or nighttime based on a weather API.

    Args:
        api_key (str): Your personal API key for the weather service.
        city (str): The name of the city to get weather data for.

    Returns:
        int: 0 if it is daytime, 1 if it is nighttime.
    """
    print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching sunrise and sunset times for {city}...")
    
    # URL for the OpenWeatherMap API's current weather data
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"

    try:
        # Make the API request
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        # Extract sunrise and sunset times from the API response
        sunrise_timestamp = data['sys']['sunrise']
        sunset_timestamp = data['sys']['sunset']
        
        # Convert timestamps to datetime objects in local time
        sunrise_time = datetime.datetime.fromtimestamp(sunrise_timestamp)
        sunset_time = datetime.datetime.fromtimestamp(sunset_timestamp)

        print(f"Sunrise: {sunrise_time.strftime('%H:%M:%S')}")
        print(f"Sunset:  {sunset_time.strftime('%H:%M:%S')}")
        
        # Get the current time
        current_time = datetime.datetime.now()
        print(f"Current Time: {current_time.strftime('%H:%M:%S')}")

        # Check if the current time is between sunrise and sunset
        if sunrise_time < current_time < sunset_time:
            print("It is daytime.")
            return 0  # Daytime
        else:
            print("It is nighttime.")
            return 1  # Nighttime

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from the API: {e}")
        return -1  # Return a value indicating an error
    except KeyError:
        print("Error: Could not parse sunrise/sunset data from the API response. Check your city name or API key.")
        return -1

if __name__ == "__main__":
    # You MUST replace 'YOUR_API_KEY' with a valid key from OpenWeatherMap
    WEATHER_API_KEY = "0b3c2a533b511a7b352a58fce05c83e1"
    
    # Location for which you want to get sunrise/sunset times
    CITY_NAME = "Kolkata"

    # Main loop to update the status every minute
    while True:
        try:
            # The function now uses the real-time, current time
            status = get_day_night_status(WEATHER_API_KEY, CITY_NAME)
            print(f"Current Day/Night Status (0=day, 1=night): {status}")

        except Exception as e:
            print(f"An error occurred: {e}")
        
        # Pause the script for 60 seconds (1 minute)
        print("Waiting 60 seconds for the next update...")
        time.sleep(60)

