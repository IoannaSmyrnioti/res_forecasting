from tinydb import TinyDB, Query
import copy
from res_forecasting import data
import importlib.resources as pkg_resources

class WeatherDataStorage:
    def __init__(self, file_path="weather_data.json"):
        with pkg_resources.path(data, file_path) as db_path:
            self.db = TinyDB(db_path)
        self.table = self.db.table("weather")

    def store_weather_data(self, data):
        
        if "days" not in data:
            raise ValueError("Invalid data: missing 'days' field")

        doc_latitude = data.get("latitude")
        doc_longitude = data.get("longitude")
        no_inserted_docs = 0
        for i in range(len(data["days"])):
            doc_date = data["days"][i].get("datetime")
            
            Weather = Query()
            query = Weather.datetime == doc_date
            query &= Weather.latitude == doc_latitude
            query &= Weather.longitude == doc_longitude

            exists = self.table.contains(query)
            if exists:
                print(f"Skipping duplicate for date {doc_date} at {doc_latitude}, {doc_longitude}")
            else:
                single_day_data = copy.deepcopy(data)
                del single_day_data["days"]
                for key in  data["days"][i].keys():
                    single_day_data[key] = data["days"][i][key] 
                self.table.insert(single_day_data)   
                no_inserted_docs += 1 
            
        print(f"Inserted {no_inserted_docs} day records into TinyDB.")

    def find_by(self, date, latitude, longitude):
        Weather = Query()
        query = Weather.datetime == date
        query &= Weather.latitude == latitude
        query &= Weather.longitude == longitude
        return self.table.search(query)
