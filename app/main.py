from dotenv import load_dotenv
load_dotenv()

import os
from datetime import datetime
from dateutil.parser import parse as parse_date

import openmeteo_requests
import requests_cache
from retry_requests import retry

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from openai import OpenAI
from app.predict import predict_pm25

# Initialize OpenAI client
client = OpenAI()

# Initialize FastAPI app
app = FastAPI(
    title="Sheffield Pollution API",
    description="API for forecasting pollution in Sheffield",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler to aid debugging
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "trace": traceback.format_exc()},
    )

@app.get("/")
async def root():
    return {"message": "Welcome to the Sheffield Pollution API"}

# Request and response models
class PollutionForecastRequest(BaseModel):
    query: str

class PollutionForecastResponse(BaseModel):
    summary: str

@app.post("/pollution-forecast")
async def pollution_forecast(request: PollutionForecastRequest) -> PollutionForecastResponse:
    query = request.query
    print("Received query:", query)

    # Extract date using OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts future dates in YYYY-MM-DD format."},
                {"role": "system", "content": f"The current datetime is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. You should use this to extract a date from the user's query."},
                {"role": "user", "content": f"Extract the date from: '{query}'"}
            ]
        )
        extracted_date_str = response.choices[0].message.content.strip()
        print("Extracted date string:", extracted_date_str)
    except Exception as e:
        print("OpenAI API error:", e)
        return JSONResponse(status_code=500, content={"error": "Failed to parse date", "details": str(e)})

    # Parse extracted date
    try:
        date_obj = parse_date(extracted_date_str)
        date = date_obj.strftime("%Y-%m-%d")
    except Exception as e:
        print("Date parsing error:", e)
        date = datetime.now().strftime("%Y-%m-%d")

    # Fetch weather data
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": 53.3790642,
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

    try:
        [response] = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        hourly = response.Hourly()
    except Exception as e:
        print("Weather API error:", e)
        return JSONResponse(status_code=500, content={"error": "Failed to fetch weather data", "details": str(e)})

    # Predict PM2.5
    try:
        prediction_vars = {
            "temperatures": hourly.Variables(0).ValuesAsNumpy(),
            "humidities": hourly.Variables(1).ValuesAsNumpy(),
            "wind_speeds": hourly.Variables(2).ValuesAsNumpy(),
            "wind_directions": hourly.Variables(3).ValuesAsNumpy(),
            "surface_pressures": hourly.Variables(4).ValuesAsNumpy(),
        }
        pm25_prediction = predict_pm25(**prediction_vars)
    except Exception as e:
        print("Prediction error:", e)
        return JSONResponse(status_code=500, content={"error": "Failed to predict PM2.5", "details": str(e)})

    # Summarise using OpenAI
    try:
        weather_summary = "\n".join(f"{k}: {v.tolist()[:3]}..." for k, v in prediction_vars.items())
        summary_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarises a prediction of PM2.5 pollution for a specific future date."},
                {"role": "system", "content": f"You were asked the question '{query}'. The date is {date}. The PM2.5 prediction is {pm25_prediction}. This is based on the weather data:\n{weather_summary}. Summarise this and explain why the weather may have contributed to the prediction."}
            ]
        )
        summary = summary_response.choices[0].message.content.strip()
    except Exception as e:
        print("Summary generation error:", e)
        summary = f"Predicted PM2.5 for {date} is {pm25_prediction}."

    return PollutionForecastResponse(summary=summary)
