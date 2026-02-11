"""
Learned Collision Detector using Trained ML Models

This module uses trained machine learning models for collision detection.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.safety.collision_data_collector import FeatureExtractor

logger = logging.getLogger(__name__)


class LearnedCollisionDetector:
    """Collision detection using trained machine learning models.

    Supports:
    - Neural network models (PyTorch)
    - Random Forest models (scikit-learn)
    - Isolation Forest (unsupervised anomaly detection)

    Usage:
        detector = LearnedCollisionDetector.load(model_path)
        result = detector.check_collision(observation, action)
        if result.is_detected:
            handle_collision(result)
    """

    def __init__(
        self,
        model,
        model_type: str,
        feature_extractor: FeatureExtractor,
        base_torques: dict[str, float],
        threshold: float = 0.5,
        device: str = "auto",
    ):
        """Initialize the learned collision detector.

        Args:
            model: Trained model (PyTorch model or sklearn classifier).
            model_type: Type of model ("nn", "rf", "isolation_forest").
            feature_extractor: Feature extractor used for training.
            base_torques: Base torque values from calibration.
            threshold: Decision threshold for binary classification (0-1).
            device: Device for PyTorch models ("auto", "cpu", "cuda").
        """
        self.model = model
        self.model_type = model_type
        self.feature_extractor = feature_extractor
        self.base_torques = base_torques
        self.threshold = threshold

        # Setup device for neural network
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model_type == "nn":
            self.model = self.model.to(self.device)
            self.model.eval()

        # Statistics
        self._total_checks = 0
        self._collision_count = 0
        self._confidence_history = []

        # Feature scaler (should be loaded from training)
        self._scaler = None

    @classmethod
    def load(cls, model_path: str, device: str = "auto") -> "LearnedCollisionDetector":
        """Load a trained model from disk.

        Args:
            model_path: Path to the model directory.
            device: Device for PyTorch models.

        Returns:
            Loaded LearnedCollisionDetector instance.
        """
        model_path = Path(model_path)

        # Load metadata
        metadata_path = model_path / "collision_model_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        model_type = metadata["model_type"]
        base_torques = metadata["base_torques"]
        fe_config = metadata.get("feature_extractor_config", {})

        # Create feature extractor
        feature_extractor = FeatureExtractor(
            history_length=fe_config.get("history_length", 10)
        )

        # Load model
        if model_type == "nn":
            model_file = model_path / "collision_model.pt"
            checkpoint = torch.load(model_file, map_location="cpu")

            from lerobot.safety.train_collision_model import CollisionDetectionNet

            model_config = checkpoint["model_config"]
            model = CollisionDetectionNet(
                input_dim=model_config["input_dim"],
                hidden_dims=model_config["hidden_dims"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])

            logger.info(f"Loaded neural network model from {model_file}")
        elif model_type == "rf":
            import joblib

            model_file = model_path / "collision_model.pkl"
            model = joblib.load(model_file)
            logger.info(f"Loaded Random Forest model from {model_file}")
        elif model_type == "isolation_forest":
            import joblib

            model_file = model_path / "collision_model.pkl"
            model = joblib.load(model_file)
            logger.info(f"Loaded Isolation Forest model from {model_file}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Try to load scaler if exists
        scaler_path = model_path / "feature_scaler.pkl"
        if scaler_path.exists():
            import joblib
            detector = cls(
                model=model,
                model_type=model_type,
                feature_extractor=feature_extractor,
                base_torques=base_torques,
                device=device,
            )
            detector._scaler = joblib.load(scaler_path)
            logger.info(f"Loaded feature scaler from {scaler_path}")

        return cls(
            model=model,
            model_type=model_type,
            feature_extractor=feature_extractor,
            base_torques=base_torques,
            device=device,
        )

    def check_collision(
        self,
        observation: dict[str, Any],
        action: dict[str, Any] | None = None,
    ) -> "CollisionResult":
        """Check if a collision has occurred.

        Args:
            observation: Current observation dict.
            action: Current action dict.

        Returns:
            CollisionResult with detection status and details.
        """
        self._total_checks += 1
        timestamp = time.time()

        # Extract features
        features = self.feature_extractor.extract_features(
            observation, action, self.base_torques
        )

        # Apply scaler if available
        if self._scaler is not None:
            features = self._scaler.transform([features])[0]
        else:
            features = np.array([features])

        # Get prediction
        confidence, severity = self._predict(features)

        # Determine collision based on threshold
        is_detected = confidence > self.threshold

        if is_detected:
            self._collision_count += 1

        # Store confidence for monitoring
        self._confidence_history.append(confidence)
        if len(self._confidence_history) > 1000:
            self._confidence_history.pop(0)

        # Create result
        result = CollisionResult(
            timestamp=timestamp,
            is_detected=is_detected,
            confidence=confidence,
            severity=severity,
            raw_features=features[0] if len(features) == 1 else features,
            detection_method=f"learned_{self.model_type}",
        )

        return result

    def _predict(self, features: np.ndarray) -> tuple[float, str]:
        """Get collision prediction from model.

        Args:
            features: Feature array (may be batched).

        Returns:
            Tuple of (confidence, severity).
        """
        if self.model_type == "nn":
            return self._predict_nn(features)
        elif self.model_type == "rf":
            return self._predict_rf(features)
        elif self.model_type == "isolation_forest":
            return self._predict_iforest(features)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _predict_nn(self, features: np.ndarray) -> tuple[float, str]:
        """Predict using neural network."""
        features_tensor = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            collision_logits, severity_logits = self.model(features_tensor)

            # Get collision probability
            collision_probs = torch.softmax(collision_logits, dim=1)
            confidence = collision_probs[0, 1].item()

            # Get severity prediction
            severity_probs = torch.softmax(severity_logits, dim=1)
            severity_idx = torch.argmax(severity_probs[0]).item()

        severity_map = {0: "none", 1: "low", 2: "medium", 3: "high", 4: "critical"}
        severity = severity_map.get(severity_idx, "unknown")

        return confidence, severity

    def _predict_rf(self, features: np.ndarray) -> tuple[float, str]:
        """Predict using Random Forest."""
        # Get probability prediction
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            confidence = probs[1] if len(probs) > 1 else probs[0]
        else:
            prediction = self.model.predict(features)[0]
            confidence = float(prediction)

        # For severity, use a simple heuristic based on confidence
        if confidence > 0.9:
            severity = "critical"
        elif confidence > 0.75:
            severity = "high"
        elif confidence > 0.6:
            severity = "medium"
        elif confidence > 0.5:
            severity = "low"
        else:
            severity = "none"

        return confidence, severity

    def _predict_iforest(self, features: np.ndarray) -> tuple[float, str]:
        """Predict using Isolation Forest."""
        # Isolation Forest returns -1 for anomalies, 1 for normal
        prediction = self.model.predict(features)[0]
        score = self.model.score_samples(features)[0]

        # Convert to confidence (0-1, higher = more likely collision)
        # Anomaly score is typically negative, so we invert and normalize
        confidence = 1.0 / (1.0 + np.exp(score))

        if prediction == -1:  # Anomaly detected
            confidence = max(confidence, 0.51)  # Ensure above threshold

        # Severity based on confidence
        if confidence > 0.9:
            severity = "critical"
        elif confidence > 0.75:
            severity = "high"
        elif confidence > 0.6:
            severity = "medium"
        elif confidence > 0.5:
            severity = "low"
        else:
            severity = "none"

        return confidence, severity

    def get_statistics(self) -> dict[str, Any]:
        """Get detection statistics."""
        avg_confidence = (
            np.mean(self._confidence_history) if self._confidence_history else 0.0
        )

        return {
            "total_checks": self._total_checks,
            "collision_count": self._collision_count,
            "collision_rate": (
                self._collision_count / self._total_checks
                if self._total_checks > 0
                else 0.0
            ),
            "average_confidence": avg_confidence,
            "model_type": self.model_type,
            "device": str(self.device),
        }

    def reset(self):
        """Reset detector state."""
        self.feature_extractor.reset()
        self._confidence_history.clear()


class CollisionResult:
    """Result of collision detection check."""

    def __init__(
        self,
        timestamp: float,
        is_detected: bool = False,
        confidence: float = 0.0,
        severity: str = "none",
        raw_features: np.ndarray | None = None,
        detection_method: str = "unknown",
    ):
        self.timestamp = timestamp
        self.is_detected = is_detected
        self.confidence = confidence
        self.severity = severity
        self.raw_features = raw_features
        self.detection_method = detection_method

    def __str__(self):
        status = "COLLISION" if self.is_detected else "OK"
        return (
            f"CollisionResult({status}, confidence={self.confidence:.3f}, "
            f"severity={self.severity}, method={self.detection_method})"
        )
