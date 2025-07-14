from res_forecasting.helpers.weather_service.weather_client import WeatherAPIClient
from res_forecasting.helpers.weather_service.storage_client import WeatherDataStorage
import requests
import os
from dotenv import load_dotenv
from tinydb.storages import MemoryStorage
from tinydb import TinyDB

# Load environment variables from .env file
load_dotenv()

def test_retrieve_and_store_data(requests_mock):

    api_client = WeatherAPIClient()

    mock_url = api_client.base_url + "/38.9697,-77.385/2020-10-01?key=" + str(os.getenv("WEATHER_API_KEY"))
    mock_response = {'queryCost': 24, 'latitude': 38.9697, 'longitude': -77.385, 'resolvedAddress': '38.9697,-77.385', 'address': '38.9697,-77.385', 'timezone': 'America/New_York', 'tzoffset': -4.0, 'days': [{'datetime': '2020-10-01', 'datetimeEpoch': 1601524800, 'tempmax': 70.7, 'tempmin': 52.7, 'temp': 61.0, 'feelslikemax': 70.7, 'feelslikemin': 52.7, 'feelslike': 61.0, 'dew': 52.6, 'humidity': 75.6, 'precip': 0.022, 'precipprob': 100.0, 'precipcover': 8.33, 'preciptype': ['rain'], 'snow': 0.0, 'snowdepth': 0.0, 'windgust': None, 'windspeed': 10.2, 'winddir': 306.0, 'pressure': 1013.5, 'cloudcover': 51.0, 'visibility': 9.9, 'solarradiation': 75.0, 'solarenergy': 6.5, 'uvindex': 4.0, 'sunrise': '07:05:56', 'sunriseEpoch': 1601550356, 'sunset': '18:51:28', 'sunsetEpoch': 1601592688, 'moonphase': 0.5, 'conditions': 'Rain, Partially cloudy', 'description': 'Partly cloudy throughout the day with late afternoon rain.', 'icon': 'rain', 'stations': ['72405503714', 'KIAD', '72403093738', 'C3816', 'KJYO', 'KGAI', 'D3839', 'F6547', '72033493764'], 'source': 'obs', 'hours': [{'datetime': '00:00:00', 'datetimeEpoch': 1601524800, 'temp': 58.4, 'feelslike': 58.4, 'humidity': 79.15, 'dew': 52.0, 'precip': 0.0, 'precipprob': 0.0, 'snow': 0.0, 'snowdepth': 0.0, 'preciptype': None, 'windgust': None, 'windspeed': 9.7, 'winddir': 190.0, 'pressure': 1012.2, 'visibility': 9.9, 'cloudcover': 0.0, 'solarradiation': 0.0, 'solarenergy': 0.0, 'uvindex': 0.0, 'conditions': 'Clear', 'icon': 'clear-night', 'stations': ['72405503714', 'KIAD', '72403093738', 'KJYO', 'KGAI', '72033493764'], 'source': 'obs'}]}], 'stations': {'72405503714': {'distance': 19167.0, 'latitude': 39.078, 'longitude': -77.557, 'useCount': 0, 'id': '72405503714', 'name': 'LEESBURG EXECUTIVE AIRPORT, VA US', 'quality': 100, 'contribution': 0.0}, 'KIAD': {'distance': 6039.0, 'latitude': 38.95, 'longitude': -77.45, 'useCount': 0, 'id': 'KIAD', 'name': 'KIAD', 'quality': 100, 'contribution': 0.0}, '72403093738': {'distance': 6640.0, 'latitude': 38.935, 'longitude': -77.447, 'useCount': 0, 'id': '72403093738', 'name': 'WASHINGTON DULLES INTERNATIONAL AIRPORT, VA US', 'quality': 100, 'contribution': 0.0}, 'C3816': {'distance': 7250.0, 'latitude': 38.938, 'longitude': -77.312, 'useCount': 0, 'id': 'C3816', 'name': 'CW3816 Reston VA US', 'quality': 0, 'contribution': 0.0}, 'KJYO': {'distance': 19489.0, 'latitude': 39.08, 'longitude': -77.56, 'useCount': 0, 'id': 'KJYO', 'name': 'KJYO', 'quality': 100, 'contribution': 0.0}, 'KGAI': {'distance': 29025.0, 'latitude': 39.17, 'longitude': -77.17, 'useCount': 0, 'id': 'KGAI', 'name': 'GAITHERSBURG, MD', 'quality': 100, 'contribution': 0.0}, 'D3839': {'distance': 7931.0, 'latitude': 39.013, 'longitude': -77.312, 'useCount': 0, 'id': 'D3839', 'name': 'DW3839 Great Falls VA US', 'quality': 0, 'contribution': 0.0}, 'F6547': {'distance': 9973.0, 'latitude': 38.881, 'longitude': -77.366, 'useCount': 0, 'id': 'F6547', 'name': 'FW6547 Oakton VA US', 'quality': 0, 'contribution': 0.0}, '72033493764': {'distance': 28929.0, 'latitude': 39.167, 'longitude': -77.167, 'useCount': 0, 'id': '72033493764', 'name': 'GAITHERSBURG MONTGOMERY CO AIR PARK, MD US', 'quality': 100, 'contribution': 0.0}}}
    requests_mock.get(mock_url, json=mock_response)
    
    data = api_client.get_weather_data(38.9697, -77.385, "2020-10-01", timeout = 10)
    assert data is not None

    storage = WeatherDataStorage()
    storage.db = TinyDB(storage=MemoryStorage)
    storage.store_weather_data(data)

    results = storage.find_by("2020-10-01", 38.9697, -77.385)
    assert len(results) == 1
    assert results[0]["datetime"] == "2020-10-01"
