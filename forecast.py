import pandas as pd
import numpy as np
import openmeteo_requests
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import import_openmeteo_requests as main

daily_dataframe = pd.read_csv('daily_weather_data.csv')

print(daily_dataframe.info())

daily_dataframe.dropna(how='all', axis=1, inplace=True)

daily_dataframe.info()
for column in daily_dataframe.columns:
    
    if daily_dataframe[column].dtype != 'datetime64[ns, UTC]':  

        mode_value = daily_dataframe[column].mode()[0]

        daily_dataframe[column].fillna(mode_value, inplace=True)

daily_dataframe['precipitation_sum'] = daily_dataframe['precipitation_sum'].apply(lambda x: 0 if x == 0 else 1).astype(int)

print("Updated precipitation_sum column:")
print(daily_dataframe[['date', 'precipitation_sum']].head())

X = daily_dataframe.drop(columns=['date', 'precipitation_sum'])
y = daily_dataframe['precipitation_sum']
X.info()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


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
for name, mp in models_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='accuracy')
    
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    results[name] = {
        'best_params': clf.best_params_,
        'accuracy': accuracy_score(y_test, y_pred)
    }
    
    print(classification_report(y_test, y_pred))

print("Summary of Results:", results)

comparison_data = []
for model_name, data in results.items():
    comparison_data.append({
        'Model': model_name,
        'Accuracy': data['accuracy'],
        'Best Parameters': str(data['best_params'])
    })

comparison_df = pd.DataFrame(comparison_data).sort_values(by='Accuracy', ascending=False)

print(f"Model Comparison Table (Sorted by Accuracy): {comparison_df}")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_params = results[best_model_name]['best_params']

print(f"Best Model: {best_model_name}")
print(f"Best Parameters: {best_params}")

if best_model_name == 'LogisticRegression':
    final_model = LogisticRegression(**best_params)
elif best_model_name == 'DecisionTree':
    final_model = DecisionTreeClassifier(**best_params)
elif best_model_name == 'RandomForest':
    final_model = RandomForestClassifier(**best_params)
elif best_model_name == 'KNN':
    final_model = KNeighborsClassifier(**best_params)

final_model.fit(X_train_scaled, y_train)

def plot_best_model_importance(model, features):
    import matplotlib.pyplot as plt
    import numpy as np
    
    model_name = type(model).__name__
    plt.figure(figsize=(10, 6))
    
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        title = f'Feature Importances: {model_name}'
        xlabel = 'Gini Importance'
    
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0]
        title = f'Feature Coefficients: {model_name}'
        xlabel = 'Coefficient Value'
    else:
        print(f'Model type {model_name} does not support standard importance plotting.')
        return

    indices = np.argsort(np.abs(importances))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel(xlabel)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_best_model_importance(final_model, X.columns)

joblib.dump(final_model, 'best_weather_model.pkl')

try:
    forecast_days = int(input("Enter number of days to forecast (1-16): ") or "7")
except ValueError:
    forecast_days = 7
    
forecast_url = "https://api.open-meteo.com/v1/forecast"

forecast_params = {
    "latitude": main.lat,
    "longitude": main.lon,
    "daily": ["temperature_2m_mean", "precipitation_probability_mean", "relative_humidity_2m_mean", "surface_pressure_mean", "visibility_mean", "dew_point_2m_mean"],
    "timezone": "auto",
    "forecast_days": forecast_days
}

forecast_responses = main.openmeteo.weather_api(forecast_url, params=forecast_params)
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

features_to_predict = df_forecast.drop(columns=['date'])

common_columns = features_to_predict.columns.intersection(X.columns)
features_to_predict_fit = features_to_predict[common_columns]

# features_to_predict_fit.info()
# X.info()

scaled_forecast = scaler.transform(features_to_predict_fit)

predictions = final_model.predict(scaled_forecast)
probabilities = final_model.predict_proba(scaled_forecast)[:, 1] if hasattr(final_model, 'predict_proba') else [None] * len(predictions)

df_forecast['Rain_Prediction'] = ["Опади очікуються" if p == 1 else "Опадів не очікується" for p in predictions]
df_forecast['Rain_Probability'] = probabilities

print(f"7-Day Forecast Analysis:")
print(df_forecast[['date', 'temperature_2m_mean', 'Rain_Prediction', 'Rain_Probability']])

