import openmeteo_requests
import pandas as pd
import requests_cache
import requests
from retry_requests import retry

city_name = input("Enter city name: ")
start_date = input("Enter start date (YYYY-MM-DD): ")
end_date = input("Enter end date (YYYY-MM-DD): ")

geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
geo_response = requests.get(geocode_url).json()

if not geo_response.get('results'):
    print(f"Could not find coordinates for {city_name}. Please check the spelling.")
else:
    location = geo_response['results'][0]
    lat, lon = location['latitude'], location['longitude']
    print(f"Found {location['name']}, {location.get('country', '')}: {lat}, {lon}")

    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_mean", "precipitation_probability_mean", "relative_humidity_2m_mean", "precipitation_sum", "surface_pressure_mean", "visibility_mean", "dew_point_2m_mean"],
        "start_date": start_date,
        "end_date": end_date,
    }
    responses = openmeteo.weather_api(url, params = params)
    
    response = responses[0]
    daily = response.Daily()
    
    daily_data = {
        "date": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
            end =  pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = daily.Interval()),
            inclusive = "left"
        )
    }
    
    variable_names = ["temperature_2m_mean", "precipitation_probability_mean", "relative_humidity_2m_mean", "precipitation_sum", "surface_pressure_mean", "visibility_mean", "dew_point_2m_mean"]
    for i, var_name in enumerate(variable_names):
        daily_data[var_name] = daily.Variables(i).ValuesAsNumpy()

    daily_dataframe = pd.DataFrame(data = daily_data)

daily_dataframe.to_csv('daily_weather_data.csv', index=False)
print("DataFrame saved successfully to 'daily_weather_data.csv'.")