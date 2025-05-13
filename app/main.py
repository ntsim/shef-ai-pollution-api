import openmeteo_requests

import requests_cache
from retry_requests import retry

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.predict import predict_pm25

app = FastAPI(
    title="Sheffield Pollution API",
    description="API for forecasting pollution in Sheffield",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Sheffield Pollution API"}

# Request and Response Models
class PollutionForecastRequest(BaseModel):
    """Forecast query in natural language e.g. 'What will the pollution be like on Tuesday?'"""
    query: str

class PollutionForecastResponse(BaseModel):
    """The PM2.5 prediction"""
    pm25: str


@app.post("/pollution-forecast")
async def pollution_forecast(request: PollutionForecastRequest) -> PollutionForecastResponse:
    """
    Create a pollution forecast.
    """

    # TODO: Parse query with OpenAI to get date
    date = "2025-05-14"

    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        # Hard coded to Sheffield City Hall
        "latitude":  53.3790642,
        "longitude": -1.4696936,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure"
        ],
        "start_date": date,
        "end_date": date,
    }

    [response] = openmeteo.weather_api(url, params=params)

    hourly = response.Hourly()

    pm25_prediction = predict_pm25(
        temperatures = hourly.Variables(0).ValuesAsNumpy(),
        humidities = hourly.Variables(1).ValuesAsNumpy(),
        wind_speeds = hourly.Variables(2).ValuesAsNumpy(),
        wind_directions = hourly.Variables(3).ValuesAsNumpy(),
        surface_pressures = hourly.Variables(4).ValuesAsNumpy(),
    )

    # TODO: Generate OpenAI response with prediction

    return PollutionForecastResponse(
        pm25 = pm25_prediction
    )



# Include routers here
# from app.api.v1 import api_router
# app.include_router(api_router, prefix="/api/v1")
