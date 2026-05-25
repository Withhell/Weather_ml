import streamlit as st
import pandas as pd
import numpy as np
import requests
import requests_cache
from retry_requests import retry
import openmeteo_requests
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import datetime


st.set_page_config(page_title="Weather Forecast Using Machine Learning", layout="wide")
st.title("Weather Forecast Using Machine Learning")


if 'daily_dataframe' not in st.session_state:
    st.session_state.daily_dataframe = None
if 'lat' not in st.session_state:
    st.session_state.lat = None
if 'lon' not in st.session_state:
    st.session_state.lon = None
if 'city_name_found' not in st.session_state:
    st.session_state.city_name_found = None
if 'final_model' not in st.session_state:
    st.session_state.final_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'classification_report' not in st.session_state:
    st.session_state.classification_report = None

# BLOCK 1: Data Retrieval
st.header("Block 1: Historical Data Retrieval")

col1, col2, col3 = st.columns(3)
with col1:
    city_input = st.text_input("Enter City/Country Name:", value="Kyiv")
with col2:
    start_date = st.date_input("Start Date:", value=datetime.date(2025, 5, 25))
with col3:
    end_date = st.date_input("End Date:", value=datetime.date(2026, 5, 25))

if st.button("Fetch Historical Weather Data", key="fetch_data_btn"):
    with st.spinner("Fetching coordinates and weather data..."):
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_input}&count=1&language=en&format=json"
        try:
            geo_response = requests.get(geocode_url).json()
            if not geo_response.get('results'):
                st.error(f"Could not find coordinates for '{city_input}'. Please check the spelling.")
            else:
                location = geo_response['results'][0]
                lat = location['latitude']
                lon = location['longitude']
                city_name_found = f"{location['name']}, {location.get('country', '')}"
                
                st.session_state.lat = lat
                st.session_state.lon = lon
                st.session_state.city_name_found = city_name_found
                                
                cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
                retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
                openmeteo = openmeteo_requests.Client(session=retry_session)
                
                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "daily": ["temperature_2m_mean", "precipitation_probability_mean", "relative_humidity_2m_mean", "precipitation_sum", "surface_pressure_mean", "visibility_mean", "dew_point_2m_mean"],
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                }
                responses = openmeteo.weather_api(url, params=params)
                response = responses[0]
                daily = response.Daily()
                
                daily_data = {
                    "date": pd.date_range(
                        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=daily.Interval()),
                        inclusive="left"
                    )
                }
                
                variable_names = ["temperature_2m_mean", "precipitation_probability_mean", "relative_humidity_2m_mean", "precipitation_sum", "surface_pressure_mean", "visibility_mean", "dew_point_2m_mean"]
                for i, var_name in enumerate(variable_names):
                    daily_data[var_name] = daily.Variables(i).ValuesAsNumpy()
                
                df = pd.DataFrame(data=daily_data)
                st.session_state.daily_dataframe = df
                st.success(f"Successfully loaded data for {city_name_found}!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if st.session_state.daily_dataframe is not None:
    st.subheader(f"Dataframe Preview ({st.session_state.city_name_found})")
    st.dataframe(st.session_state.daily_dataframe)
        
    csv_data = st.session_state.daily_dataframe.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Save DataFrame as CSV",
        data=csv_data,
        file_name='daily_weather_data.csv',
        mime='text/csv',
    )


# BLOCK 2: Model Training
st.header("Block 2: Model Training & Evaluation")

if st.session_state.daily_dataframe is None:
    st.info("Please fetch historical data in Block 1 before training models.")
else:
    if st.button("Train ML Models", key="train_models_btn"):
        with st.spinner("Preprocessing data and optimization via GridSearchCV..."):
            df_train = st.session_state.daily_dataframe.copy()
            
            df_train.dropna(how='all', axis=1, inplace=True)
            
            for column in df_train.columns:
                if df_train[column].dtype != 'datetime64[ns, UTC]':
                    mode_value = df_train[column].mode()[0]
                    df_train[column] = df_train[column].fillna(mode_value)
                        
            df_train['precipitation_sum'] = df_train['precipitation_sum'].apply(lambda x: 0 if x == 0 else 1).astype(int)
            
            X = df_train.drop(columns=['date', 'precipitation_sum'])
            y = df_train['precipitation_sum']
            
            st.session_state.X_columns = X.columns.tolist()
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.session_state.scaler = scaler
            
            models_params = {
                'LogisticRegression': {
                    'model': LogisticRegression(),
                    'params': {'C': [0.1, 1, 10]}
                },
                'DecisionTree': {
                    'model': DecisionTreeClassifier(),
                    'params': {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
                },
                'RandomForest': {
                    'model': RandomForestClassifier(),
                    'params': {'n_estimators': [50, 100], 'max_depth': [None, 5]}
                },
                'KNN': {
                    'model': KNeighborsClassifier(),
                    'params': {'n_neighbors': [3, 5, 7]}
                }
            }
            
            results = {}
            reports = {}
            for name, mp in models_params.items():
                clf = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='accuracy')
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)
                
                results[name] = {
                    'best_params': clf.best_params_,
                    'accuracy': accuracy_score(y_test, y_pred)
                }
                reports[name] = classification_report(y_test, y_pred)
            
            best_model_name = max(results, key=lambda x: results[x]['accuracy'])
            best_params = results[best_model_name]['best_params']
            
            if best_model_name == 'LogisticRegression':
                final_model = LogisticRegression(**best_params)
            elif best_model_name == 'DecisionTree':
                final_model = DecisionTreeClassifier(**best_params)
            elif best_model_name == 'RandomForest':
                final_model = RandomForestClassifier(**best_params)
            elif best_model_name == 'KNN':
                final_model = KNeighborsClassifier(**best_params)
                
            final_model.fit(X_train_scaled, y_train)
            
            st.session_state.final_model = final_model
            st.session_state.best_model_name = best_model_name
            st.session_state.classification_report = reports[best_model_name]

    if st.session_state.final_model is not None:
        st.success(f"**Best Performing Model:** {st.session_state.best_model_name}")
        
        st.subheader("Classification Report")
        st.code(st.session_state.classification_report)
        
        st.subheader("Feature Importance Analysis")
        model = st.session_state.final_model
        features = st.session_state.X_columns
        model_name = type(model).__name__
        
        fig, ax = plt.subplots(figsize=(10, 4))
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            title = f'Feature Importances: {model_name}'
            xlabel = 'Gini Importance'
        elif hasattr(model, 'coef_'):
            importances = model.coef_[0]
            title = f'Feature Coefficients: {model_name}'
            xlabel = 'Coefficient Value'
        else:
            importances = None
            st.write(f"Model type {model_name} does not support standard importance plotting.")
            
        if importances is not None:
            indices = np.argsort(np.abs(importances))
            ax.set_title(title)
            ax.barh(range(len(indices)), importances[indices], color='teal', align='center')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([features[i] for i in indices])
            ax.set_xlabel(xlabel)
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)

# BLOCK 3: Weather Forecasting
st.header("Block 3: Future Weather Forecast & Inference")

if st.session_state.final_model is None:
    st.info("Please train the models in Block 2 before generating a forecast.")
else:
    forecast_days = st.number_input("Enter number of days to forecast (1-16):", min_value=1, max_value=16, value=7)
    
    if st.button("Start Forecast", key="start_forecast_btn"):
        with st.spinner("Fetching real-time forecast and computing predictions..."):
            forecast_url = "https://api.open-meteo.com/v1/forecast"
                        
            forecast_params = {
                "latitude": st.session_state.lat,
                "longitude": st.session_state.lon,
                "daily": ["temperature_2m_mean", "precipitation_probability_mean", "relative_humidity_2m_mean", "surface_pressure_mean", "visibility_mean", "dew_point_2m_mean"],
                "timezone": "auto",
                "forecast_days": int(forecast_days)
            }
            
            try:
                cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
                retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
                openmeteo = openmeteo_requests.Client(session=retry_session)
                
                forecast_responses = openmeteo.weather_api(forecast_url, params=forecast_params)
                res = forecast_responses[0]
                d_forecast = res.Daily()
                
                forecast_data = {
                    "date": pd.date_range(
                        start=pd.to_datetime(d_forecast.Time(), unit="s", utc=True),
                        end=pd.to_datetime(d_forecast.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=d_forecast.Interval()),
                        inclusive="left"
                    )
                }
                
                forecast_vars = ["temperature_2m_mean", "precipitation_probability_mean", "relative_humidity_2m_mean", "surface_pressure_mean", "visibility_mean", "dew_point_2m_mean"]
                for i, var_name in enumerate(forecast_vars):
                    forecast_data[var_name] = d_forecast.Variables(i).ValuesAsNumpy()
                
                df_forecast = pd.DataFrame(data=forecast_data)
               
                features_to_predict = df_forecast[st.session_state.X_columns]
               
                scaled_forecast = st.session_state.scaler.transform(features_to_predict)
                
                predictions = st.session_state.final_model.predict(scaled_forecast)
                probabilities = st.session_state.final_model.predict_proba(scaled_forecast)[:, 1] if hasattr(st.session_state.final_model, 'predict_proba') else [np.nan] * len(predictions)
                
                df_forecast['Rain_Prediction'] = ["Опади очікуються" if p == 1 else "Опадів не очікується" for p in predictions]
                df_forecast['Rain_Probability'] = probabilities
                
                df_forecast['date'] = df_forecast['date'].dt.strftime('%Y-%m-%d')
                
                st.subheader(f"{forecast_days}-Day Forecast Results ({st.session_state.city_name_found})")
                st.dataframe(df_forecast[['date', 'Rain_Prediction', 'Rain_Probability']])
                
                with st.expander("View all parsed incoming meteorological feature metrics"):
                    st.dataframe(df_forecast)
                    
            except Exception as e:
                st.error(f"An error occurred during forecasting: {e}")