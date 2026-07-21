from __future__ import annotations

import os
import pickle
import logging
from typing import Any

import numpy as np

from prediction.features import engineer_features, FEATURE_ORDER
from prediction.repository import PredictionRepository
from prediction.service import PredictionService

logger = logging.getLogger("factorymind")


class PredictionEngine:
    def __init__(self):
        self.repository = PredictionRepository()
        self.service = PredictionService(self.repository)
        self.model = None
        self.scaler = None
        self.load_model()
        logger.info("PredictionEngine initialized.")

    def load_model(self):
        """Load the trained XGBoost model dict from disk."""
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "model", "xgboost_model.pkl"
        )
        if not os.path.exists(model_path):
            logger.warning(f"XGBoost model file not found at: {model_path}")
            self.model = None
            self.scaler = None
            return

        try:
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)

            if isinstance(bundle, dict):
                self.model = bundle  # keep the whole dict
                self.scaler = bundle.get("scaler")
                binary = bundle.get("binary_model")
                n_expected = getattr(binary, "n_features_in_", "?")
                logger.info(
                    f"XGBoost model bundle loaded: keys={list(bundle.keys())}, "
                    f"binary expects {n_expected} features."
                )
            else:
                # Bare model (legacy)
                self.model = {"binary_model": bundle}
                self.scaler = None
                logger.info(f"XGBoost bare model loaded, type={type(bundle)}")

        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}", exc_info=True)
            self.model = None
            self.scaler = None

    def predict(
        self,
        air_temp: float,
        process_temp: float,
        rotational_speed: float,
        torque: float,
        tool_wear: float,
        machine_type: str = "M",
    ) -> dict[str, Any]:
        """
        Run the trained XGBoost prediction pipeline.

        Returns a dict with:
          failure_probability, failure_type, confidence, explanation, telemetry
        """
        sensor_values = {
            "air_temperature": air_temp,
            "process_temperature": process_temp,
            "rotational_speed": rotational_speed,
            "torque": torque,
            "tool_wear": tool_wear,
            "vibration": 0.08,
        }

        failure_probability = 0.0
        failure_type = "None"
        confidence = 0.85

        if self.model is not None:
            try:
                binary_model = self.model.get("binary_model") if isinstance(self.model, dict) else self.model
                multiclass_model = self.model.get("multiclass_model") if isinstance(self.model, dict) else None

                # --- Build 9-feature matrix ---
                features = engineer_features(
                    air_temp, process_temp, rotational_speed, torque, tool_wear, machine_type
                )

                # Apply scaler if present (must match training pipeline)
                if self.scaler is not None:
                    features = self.scaler.transform(features).astype(np.float32)

                # Assertion guard — catches future drift immediately
                if binary_model is not None:
                    expected = binary_model.n_features_in_
                    assert features.shape[1] == expected, (
                        f"Feature mismatch: model expects {expected} features, "
                        f"got {features.shape[1]}. Check prediction/features.py."
                    )

                # --- Binary failure prediction ---
                if binary_model is not None:
                    if hasattr(binary_model, "predict_proba"):
                        probs = binary_model.predict_proba(features)
                        failure_probability = float(probs[0][1]) if probs.shape[1] > 1 else float(probs[0][0])
                    else:
                        failure_probability = float(binary_model.predict(features)[0])

                # --- Multiclass failure type ---
                CLASS_MAP = {
                    0: "None",
                    1: "Heat Dissipation Failure",
                    2: "Power Failure",
                    3: "Overstrain Failure",
                    4: "Tool Wear Failure",
                    5: "Random Failure",
                }
                if multiclass_model is not None:
                    raw_class = multiclass_model.predict(features)[0]
                    failure_type = CLASS_MAP.get(int(raw_class), str(raw_class))
                elif failure_probability > 0.5:
                    failure_type = "Generic Failure"

                # Confidence: higher when prediction is decisive (far from 0.5)
                confidence = round(min(0.99, 0.50 + abs(failure_probability - 0.5)), 4)

            except AssertionError as ae:
                logger.error(f"Feature shape assertion failed: {ae}")
                failure_type = "Feature Shape Mismatch"
            except Exception as e:
                logger.error(f"XGBoost inference failed: {e}", exc_info=True)
                failure_type = "Inference Error"
        else:
            # Heuristic fallback when model is unavailable
            logger.warning("No XGBoost model loaded — using heuristic fallback.")
            if torque > 65.0 or tool_wear > 200 or rotational_speed > 2500:
                failure_probability = 0.85
                failure_type = "Overstrain Failure" if torque > 65 else "Tool Wear Failure"
            else:
                failure_probability = 0.05
                failure_type = "None"
            confidence = 0.60

        status = (
            f"Warning: Machine has a {failure_probability * 100:.1f}% probability of failure "
            f"({failure_type})."
            if failure_probability > 0.5
            else "Operational: Machine is running within normal limits."
        )

        return {
            "failure_probability": round(failure_probability, 4),
            "failure_type": failure_type,
            "confidence": confidence,
            "explanation": status,
            "telemetry": sensor_values,
        }


prediction_engine = PredictionEngine()
