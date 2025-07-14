"""A Class to fetch weather data from Visual Crossing API."""

import os
import logging
import requests
from res_forecasting.weather_service.storage_client import WeatherDataStorage

logging.basicConfig(
    encoding="utf-8",
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class WeatherAPIClient:
    """A Class to fetch weather data from Visual Crossing API."""

    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing API key. Set WEATHER_API_KEY in .env or environment."
            )
        self.base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

    def get_weather_data(
        self, latitude, longitude, start_date, end_date=None, timeout=None, **params
    ):
        """
        Fetches weather data from Visual Crossing API.

        :param latitude: Latitude of the location
        :param longitude: Longitude of the location
        :param start_date: Start date in format YYYY-MM-DD
        :param end_date: End date in format YYYY-MM-DD
        :param params: Optional additional query parameters
        :return: Parsed JSON response or raises an exception
        """
        if end_date is None:
            url = f"{self.base_url}/{latitude},{longitude}/{start_date}"
        else:
            url = f"{self.base_url}/{latitude},{longitude}/{start_date}/{end_date}"
        params["key"] = self.api_key

        response = requests.get(url, params=params, timeout=timeout)

        try:
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            print(f"Response content: {response.text}")
            raise
        except requests.RequestException as e:
            print(f"Error occurred: {e}")
            raise


if __name__ == "__main__":
    api_client = WeatherAPIClient()
    # data = api_client.get_weather_data(38.9697, -77.385, "2020-10-01", end_date = "2020-10-02", timeout = 10)

    storage = WeatherDataStorage()
    # storage.store_weather_data(data)

    results = storage.find_by("2020-10-02", 38.9697, -77.385)
    print()
