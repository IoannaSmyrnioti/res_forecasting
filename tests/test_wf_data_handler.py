from res_forecasting.helpers.wf_data_handler import WFDataHandler
import duckdb
import os
import datetime


def test_wf_data_handler():
    
    normalize_names = True
    header = 9
    data_folder = os.path.join("tests", "test_data", "wf", "turbine_data")
    columns = [
        "# Date and time",
        "Wind speed (m/s)",
        "Wind speed Sensor 1 (m/s)",
        "Wind speed Sensor 2 (m/s)",
        "Density adjusted wind speed (m/s)",
        "Wind direction (°)",
        "Nacelle position (°)",
        "Power (kW)",
        "Potential power default PC (kW)",
        "Potential power learned PC (kW)",
        "Turbine Power setpoint (kW)",
        "Nacelle ambient temperature (°C)",
        "Ambient temperature (converter) (°C)",
        "Capacity factor",
        "Data Availability",
        "Blade angle (pitch position) A (°)",
        "Blade angle (pitch position) B (°)",
        "Blade angle (pitch position) C (°)",
        "Yaw bearing angle (°)",
        "Cable windings from calibration point",
    ]

    with WFDataHandler("wf_data.duckdb") as wf_db_handler:
        wf_db_handler.connection = duckdb.connect(database=":memory:")
        assert wf_db_handler.connection is not None
        
        wf_db_handler.ingest_directory(data_folder, columns = columns, normalize_names = normalize_names, header = header)

        df = wf_db_handler.get_df(
            site_name="Penmanshiel",
            turbine_id=1,
            datetime_start=datetime.datetime(2020, 1, 1),
            datetime_end=datetime.datetime(2022, 1, 1),
        )
        assert df is not None
        assert len(df) == 2