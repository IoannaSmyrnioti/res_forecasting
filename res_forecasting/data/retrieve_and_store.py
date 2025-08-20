from res_forecasting.helpers.weather_service.weather_client import WeatherAPIClient
from res_forecasting.helpers.weather_service.storage_client import WeatherDataStorage

def retrieve_and_store_weather_data(latitude, longitude, start_date, end_date=None, timeout=None, **params):
    """
    Retrieve weather data from the API and store it in the database.

    :param latitude: Latitude of the location
    :param longitude: Longitude of the location
    :param start_date: Start date in format YYYY-MM-DD
    :param end_date: End date in format YYYY-MM-DD (optional)
    :param timeout: Timeout for the API request (optional)
    :param params: Additional parameters for the API request
    """
    
    api_client = WeatherAPIClient()
    weather_data = api_client.get_weather_data(latitude, longitude, start_date, end_date, timeout, **params)

    storage = WeatherDataStorage("turbine1.json")
    storage.store_weather_data(weather_data) # Assuming `weather_data` is a list of dictionaries

if __name__ == "__main__":
    # Example usage
    retrieve_and_store_weather_data(
        latitude=37.7749,
        longitude=-122.4194,
        start_date="2023-10-02",
        end_date="2023-10-02",
        timeout=10,
        
    )

