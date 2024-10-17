
from fastapi import FastAPI, HTTPException
import requests
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from geopy.geocoders import Nominatim
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load pre-trained models
try:
    rf_regressor = joblib.load("./my_random_forestRegessor.joblib")
    rf_classifier = joblib.load("./my_random_forestClassifier.joblib")
    scaler = joblib.load("./standard_scaler.joblib")
    le = joblib.load("./label_encoder_soil.joblib")
    le0 = joblib.load("./label_encoder_crop.joblib")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model files not found.")

def Get_Temp(TS, T2M, T2MWET, T2M_MAX, T2M_MIN):
    # Averaging all temperature parameters
    return ((T2M_MAX + T2M_MIN) / 2 + TS + T2M + T2MWET) / 4

def Get_WindSpeed(WS2M, WS10M):
    # Average wind speed at 2m and 10m
    return (WS2M + WS10M) / 2

def Get_Aridity(PRECTOTCORR_SUM):
    # Aridity as a fraction of total precipitation
    return PRECTOTCORR_SUM / 160

def Get_SoilPH(Aridity):
    # Ensure Aridity is greater than 0.79 to avoid math errors
    if Aridity > 0.79:
        return 6.44 + 0.68 * np.log(Aridity - 0.79)
    else:
        return 6.44  # Return default or adjusted value if Aridity too low

def Get_Humidity(RH2M):
    # Custom logic for calculating humidity
    return RH2M 

def Get_SoilType(Aridity):
    # Reordered and adjusted soil type logic to avoid overlaps
    if Aridity <= 0.05:
        return 'Peaty'
    elif 0.05 < Aridity <= 0.2:
        return 'Clay'
    elif 0.5 < Aridity <= 0.7:
        return 'Saline'
    elif 0.2 < Aridity <= 0.5:
        return 'Loamy'
    else:  # Aridity > 0.7
        return 'Sandy'

def Get_SoilQuality(SoilType):
    # Return soil quality based on soil type
    soil_quality_map = {
        'Peaty': 24,
        'Loamy': 64,
        'Sandy': 38,
        'Saline': 15,
        'Clay': 47
    }
    return soil_quality_map.get(SoilType, 47)  # Default to 47 if not found



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_latitude_longitude():
    response = requests.get('https://ipinfo.io/')
    data = response.json()
    
    if 'loc' in data:
        loc = data['loc'].split(',')
        latitude = float(loc[0])
        longitude = float(loc[1])
        return latitude, longitude
    else:
        return None, None
    
# Helper function to get NASA climate data and transform it
def fetch_data(start_year: int, end_year: int, latitude: float, longitude: float):
    API_URL = f"https://power.larc.nasa.gov/api/temporal/monthly/point?start={start_year}&end={end_year}&latitude={latitude}&longitude={longitude}&community=ag&parameters=PS,TS,T2M,QV2M,RH2M,WS2M,WS10M,T2MWET,GWETTOP,T2M_MAX,T2M_MIN,GWETPROF,GWETROOT,PRECTOTCORR,ALLSKY_SRF_ALB,PRECTOTCORR_SUM,ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,ALLSKY_SFC_PAR_TOT,CLRSKY_SFC_PAR_TOT&format=csv&header=false"
    
    response = requests.get(API_URL)
    filename = 'MyLocal_Data.csv'
    with open(filename,'wb') as file:
        file.write(response.content)
    
    # Read and reshape the data
    df = pd.read_csv(filename)
    melted_df = pd.melt(df, id_vars=['PARAMETER', 'YEAR'], 
                        value_vars=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], 
                        var_name='MONTH', value_name='VALUE')
    df = melted_df.pivot_table(index=['YEAR', 'MONTH'], columns='PARAMETER', values='VALUE').reset_index()

    # Sorting by month
    month_order = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    month_mapping = {month: index for index, month in enumerate(month_order)}
    df['MONTH_NUM'] = df['MONTH'].map(month_mapping)

    # Group and sort by MONTH
    df_out = df.groupby('MONTH').mean().reset_index().sort_values(by='MONTH_NUM').drop(columns='MONTH_NUM')
    
    # Applying custom functions for transformations
    df_out['Aridity'] = df_out['PRECTOTCORR_SUM'].apply(Get_Aridity)
    df_out['Soil_Type'] = df_out['Aridity'].apply(Get_SoilType)
    df_out['Soil_pH'] = df_out['Aridity'].apply(Get_SoilPH)
    df_out['Temperature'] = df_out.apply(lambda row: Get_Temp(row['TS'], row['T2M'], row['T2MWET'], row['T2M_MAX'], row['T2M_MIN']), axis=1)
    df_out['Wind_Speed'] = df_out.apply(lambda row: Get_WindSpeed(row['WS2M'], row['WS10M']), axis=1)
    df_out['Soil_Quality'] = df_out['Soil_Type'].apply(Get_SoilQuality)
    df_out['Humidity'] = df_out.apply(lambda row: Get_Humidity(row['RH2M']), axis=1)
    df_out['N'] = len(df_out) * [66]  # Average N in soil
    df_out['P'] = len(df_out) * [53]  # Average P
    df_out['K'] = len(df_out) * [42]  # Average K

    # Dropping unnecessary columns
    df_out = df_out.drop(columns=['Aridity'])

    # Final aggregation (mean for numeric, mode for categorical)
    numeric_columns = df_out.select_dtypes(include=['float64', 'int64']).columns
    mean_values = df_out[numeric_columns].mean()
    categorical_columns = df_out.select_dtypes(include=['object']).columns
    mode_values = df_out[categorical_columns].mode().iloc[0]
    final_result = pd.concat([mean_values, mode_values])
    
    return pd.DataFrame([final_result]),response.status_code

# Endpoint to get geolocation
@app.get("/geolocation")
def fetch_geolocation():
    latitude, longitude = get_latitude_longitude()
    if latitude and longitude:
        return {"latitude": latitude, "longitude": longitude}
    return {"error": "Unable to fetch your location"}

# Endpoint to retrieve NASA climate data
@app.get("/climate-data/")
def fetch_climate_data(start_year: int, end_year: int, latitude: float, longitude: float):
    df_out, status = fetch_data(start_year, end_year, latitude, longitude)
    
    if status == 200:
        # Select only the last 10 attributes for visualization
        result = df_out[['Soil_pH', 'Temperature', 'Wind_Speed', 'Soil_Quality', 
                         'Humidity', 'N', 'P', 'K', 'Soil_Type']].to_dict(orient="records")
        
        return {"data": result}
    
    return {"error": "Failed to fetch climate data"}

# Prediction endpoint (with transformation)
@app.post("/predict/")
def predict_climate(data: dict):
    # Fetch and transform data
    transformed_data = fetch_climate_data(data['start_year'], data['end_year'], data['latitude'], data['longitude'])
    transformed_data = pd.DataFrame([transformed_data['data'][0]])
    transformed_data = transformed_data[['Soil_Type', 'Soil_pH', 'Temperature', 'Humidity', 'Wind_Speed', 'N',
       'P', 'K', 'Soil_Quality']]
    transformed_data['Soil_Type'] = le.fit_transform(transformed_data['Soil_Type'])
    new_data_scaled = scaler.transform(transformed_data)
    #Assuming the model expects features in a specific format
    yield_pred = rf_regressor.predict(new_data_scaled)
    new_data_with_yield = np.column_stack((new_data_scaled, yield_pred))
    crop_pred = rf_classifier.predict(new_data_with_yield)
    L = ['Bareley','Corn','Cotton','Potato','Rice','Soybean','Sugarcane','Sunflower','Tomato','Wheat']
    return {"Best Crop": L[le0.inverse_transform(crop_pred).tolist()[0]],"Predicted Yield":yield_pred[0]}

