import pandas as pd
import joblib
import numpy as np
import numpy.typing as npt


def predict_pm25(
    temperatures: npt.NDArray,
    humidities: npt.NDArray,
    wind_speeds: npt.NDArray,
    wind_directions: npt.NDArray,
    surface_pressures: npt.NDArray
) -> str:
    # === Step 1: Load the trained model ===
    model = joblib.load('app/pm25_model_hgboost.pkl')

    # === Step 2: Define your input data ===
    # Replace these values with real or simulated ones
    sample_input = pd.DataFrame([{
        'lag_pm25_1d': np.nan,
        'lag_pm25_2d': np.nan,
        'lag_pm10_1d': np.nan,
        'temperature_c': _get_daytime_average(temperatures),
        'humidity_pct': _get_daytime_average(humidities),
        'wind_speed_kmph': _get_daytime_average(wind_speeds),
        'wind_direction_deg': _get_daytime_average(wind_directions),
        'surface_pressure_hpa': _get_daytime_average(surface_pressures)
    }])

    # === Step 3: Make a prediction ===
    prediction = model.predict(sample_input)

    # === Step 4: Output the result ===
    result = f"{prediction[0]:.2f} µg/m³"

    print(f"Predicted PM2.5: {result}")

    return result


def _get_daytime_average(values: npt.NDArray):
    # Average between 7AM and 7PM
    # TODO: Is this reasonable?
    return values[7:19].sum() / 12
