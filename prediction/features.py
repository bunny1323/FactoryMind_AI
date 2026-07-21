from __future__ import annotations

import numpy as np


# The 9-feature pipeline used by the trained XGBoost model.
# Features are derived from 5 raw AI4I sensor readings.
# MUST match exactly what was used during model training.

FEATURE_ORDER = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_diff",
    "power",
    "torque_wear",
    "type_encoded",
]

# Machine type encoding used during training (AI4I dataset convention)
TYPE_ENCODING = {"L": 0, "M": 1, "H": 2}


def engineer_features(
    air_temp: float,
    process_temp: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    machine_type: str = "M",
) -> np.ndarray:
    """
    Build the 9-feature engineered matrix that matches the trained XGBoost model.

    Derived features:
      - temp_diff      : process_temp - air_temp
      - power          : rotational_speed * torque  (proxy for mechanical power)
      - torque_wear    : torque * tool_wear          (combined stress indicator)
      - type_encoded   : integer encoding of machine type (L=0, M=1, H=2)

    Returns shape (1, 9) float32 numpy array.
    """
    temp_diff = process_temp - air_temp
    power = rotational_speed * torque
    torque_wear = torque * tool_wear
    type_enc = float(TYPE_ENCODING.get(machine_type.upper(), 1))  # default M=1

    features = np.array(
        [[air_temp, process_temp, rotational_speed, torque, tool_wear,
          temp_diff, power, torque_wear, type_enc]],
        dtype=np.float32,
    )
    assert features.shape == (1, 9), f"Feature engineering error: expected shape (1,9), got {features.shape}"
    return features
