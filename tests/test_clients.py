from res_forecasting.weather_service.weather_client import WeatherAPIClient
from res_forecasting.weather_service.storage_client import WeatherDataStorage


def test_weather_client():

    api_client = WeatherAPIClient()
    # data = api_client.get_weather_data(38.9697, -77.385, "2020-10-01", end_date = "2020-10-02", timeout = 10)
    # assert data is not None


def test_storage_client():

    storage = WeatherDataStorage()
    results = storage.find_by("2020-10-02", 38.9697, -77.385)
    assert len(results) == 1
    assert results[0]["datetime"] == "2020-10-02"


def test_retrieve_and_store_data():

    api_client = WeatherAPIClient()
    # data = api_client.get_weather_data(38.9697, -77.385, "2020-10-01", end_date = "2020-10-02", timeout = 10)
    # assert data is not None

    storage = WeatherDataStorage()
    # storage.store_weather_data(data)

    results = storage.find_by("2020-10-02", 38.9697, -77.385)
    assert len(results) == 1
    assert results[0]["datetime"] == "2020-10-02"
