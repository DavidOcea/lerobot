"""
Train Learned Collision Detection Model

This script trains a machine learning model for collision detection
using collected robot data.
"""

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

from lerobot.safety.collision_data_collector import CollisionDataSample, FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training collision detection model."""

    # Data
    data_dir: str = "./collision_data"
    test_split: float = 0.2
    val_split: float = 0.1

    # Features
    history_length: int = 10
    normalize_features: bool = True

    # Model
    model_type: str = "nn"  # "nn", "rf", "isolation_forest"
    hidden_dims: list = None
    dropout: float = 0.2

    # Training
    batch_size: int = 256
    learning_rate: float = 0.001
    num_epochs: int = 50
    early_stop_patience: int = 5

    # Class weights (for imbalanced data)
    collision_weight: float = 5.0

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]


class CollisionDataset(torch.utils.data.Dataset):
    """PyTorch dataset for collision detection."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        severity_labels: np.ndarray = None,
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.severity_labels = (
            torch.LongTensor(severity_labels) if severity_labels is not None else None
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.severity_labels is not None:
            return self.features[idx], self.labels[idx], self.severity_labels[idx]
        return self.features[idx], self.labels[idx]


class CollisionDetectionNet(nn.Module):
    """Neural network for collision detection."""

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        # Binary classification output (collision / no collision)
        self.classifier = nn.Linear(prev_dim, 2)

        # Optional: severity regression output
        self.severity_head = nn.Linear(prev_dim, 5)  # 5 severity levels

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        features = self.network(x)
        collision_logits = self.classifier(features)
        severity_logits = self.severity_head(features)
        return collision_logits, severity_logits


def load_collision_data(data_dir: str) -> list[dict]:
    """Load collision data from JSONL files.

    Args:
        data_dir: Directory containing collision data files.

    Returns:
        List of data samples.
    """
    data_path = Path(data_dir)
    samples = []

    for file_path in data_path.glob("*.jsonl"):
        logger.info(f"Loading data from {file_path}")
        with open(file_path, "r") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue

    logger.info(f"Loaded {len(samples)} samples total")
    return samples


def prepare_training_data(
    samples: list[dict],
    feature_extractor: FeatureExtractor,
    num_joints: int = 16,
) -> tuple:
    """Prepare training data from collected samples.

    Args:
        samples: List of data samples.
        feature_extractor: Feature extractor instance.
        num_joints: Number of robot joints.

    Returns:
        Tuple of (features, labels, severity_labels, base_torques).
    """
    features_list = []
    labels = []
    severity_labels = []

    # Severity mapping to integers
    severity_map = {
        "none": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }

    # First pass: compute base torques (mean forces from non-collision samples)
    force_samples = []
    for sample in samples:
        if not sample.get("is_collision", False):
            obs = sample["observation"]
            forces = {}
            for key, value in obs.items():
                if ".force" in key:
                    joint_name = key.replace(".force", "")
                    forces[joint_name] = float(value)
            if forces:
                force_samples.append(forces)

    # Compute base torques
    base_torques = {}
    if force_samples:
        all_joints = set()
        for f in force_samples:
            all_joints.update(f.keys())

        for joint in all_joints:
            values = [f.get(joint, 0) for f in force_samples if joint in f]
            if values:
                base_torques[joint] = float(np.mean(values))

    logger.info(f"Computed base torques for {len(base_torques)} joints")

    # Second pass: extract features
    for sample in samples:
        obs = sample["observation"]
        action = sample.get("action")

        # Reset and extract features
        feature_extractor.reset()
        features = feature_extractor.extract_features(obs, action, base_torques)

        features_list.append(features)
        labels.append(1 if sample.get("is_collision", False) else 0)

        severity = sample.get("collision_severity", "none")
        severity_labels.append(severity_map.get(severity, 0))

    features_array = np.array(features_list, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.int64)
    severity_array = np.array(severity_labels, dtype=np.int64)

    logger.info(f"Prepared {len(labels_array)} samples")
    logger.info(f"Collision samples: {sum(labels_array)} ({sum(labels_array)/len(labels_array)*100:.1f}%)")
    logger.info(f"Normal samples: {len(labels_array) - sum(labels_array)} ({(1-sum(labels_array)/len(labels_array))*100:.1f}%)")

    return features_array, labels_array, severity_array, base_torques


def train_neural_network(
    config: TrainingConfig,
    train_dataset: CollisionDataset,
    val_dataset: CollisionDataset,
    input_dim: int,
) -> CollisionDetectionNet:
    """Train neural network model.

    Args:
        config: Training configuration.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        input_dim: Input feature dimension.

    Returns:
        Trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Create model
    model = CollisionDetectionNet(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)

    # Loss functions
    collision_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, config.collision_weight]).to(device)
    )
    severity_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            if len(batch) == 3:
                features, labels, severities = batch
            else:
                features, labels = batch
                severities = None

            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()

            collision_logits, severity_logits = model(features)

            loss = collision_criterion(collision_logits, labels)

            # Add severity loss if available and applicable
            if severities is not None:
                severities = severities.to(device)
                # Only compute severity loss for collision samples
                collision_mask = labels > 0
                if collision_mask.sum() > 0:
                    severity_loss = severity_criterion(
                        severity_logits[collision_mask],
                        severities[collision_mask]
                    )
                    loss += 0.1 * severity_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(collision_logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    features, labels, severities = batch
                else:
                    features, labels = batch

                features, labels = features.to(device), labels.to(device)

                collision_logits, _ = model(features)
                loss = collision_criterion(collision_logits, labels)

                val_loss += loss.item()
                _, predicted = torch.max(collision_logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        logger.info(
            f"Epoch {epoch+1}/{config.num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}"
        )

        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model_state)
    return model


def train_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
) -> RandomForestClassifier:
    """Train Random Forest classifier.

    Args:
        features: Feature array.
        labels: Label array.

    Returns:
        Trained classifier.
    """
    logger.info("Training Random Forest classifier...")

    # Use class weights to handle imbalance
    class_weights = {0: 1.0, 1: 5.0}

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    clf.fit(features, labels)
    return clf


def evaluate_model(model, features: np.ndarray, labels: np.ndarray, model_type: str = "nn"):
    """Evaluate model performance.

    Args:
        model: Trained model.
        features: Test features.
        labels: True labels.
        model_type: Type of model ("nn" or "rf").

    Returns:
        Dictionary of evaluation metrics.
    """
    if model_type == "nn":
        device = next(model.parameters()).device
        model.eval()
        features_tensor = torch.FloatTensor(features).to(device)

        with torch.no_grad():
            collision_logits, _ = model(features_tensor)
            _, predictions = torch.max(collision_logits, 1)
            predictions = predictions.cpu().numpy()
    else:
        predictions = model.predict(features)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    metrics = {
        "accuracy": float(np.mean(predictions == labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }

    # Print report
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info("\nConfusion Matrix:")
    logger.info(f"                Predicted")
    logger.info(f"              Normal  Collision")
    logger.info(f"Actual Normal   {cm[0, 0]:4d}     {cm[0, 1]:4d}")
    logger.info(f"       Collision {cm[1, 0]:4d}     {cm[1, 1]:4d}")
    logger.info("=" * 60)

    return metrics


def save_model(model, output_path: str, feature_extractor, base_torques: dict, model_type: str = "nn"):
    """Save trained model to disk.

    Args:
        model: Trained model.
        output_path: Path to save the model.
        feature_extractor: Feature extractor used for training.
        base_torques: Base torques used for feature extraction.
        model_type: Type of model.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if model_type == "nn":
        model_path = output_path / "collision_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": model.network[0].in_features,
                "hidden_dims": [layer.out_features for layer in model.network if isinstance(layer, nn.Linear)],
            },
        }, model_path)
        logger.info(f"Saved neural network model to {model_path}")
    else:
        import joblib
        model_path = output_path / "collision_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved Random Forest model to {model_path}")

    # Save metadata
    metadata = {
        "model_type": model_type,
        "base_torques": base_torques,
        "feature_extractor_config": {
            "history_length": feature_extractor.history_length,
        },
    }

    metadata_path = output_path / "collision_model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Train collision detection model")
    parser.add_argument("--data-dir", type=str, default="./collision_data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./models/collision_detection", help="Output directory")
    parser.add_argument("--model-type", type=str, default="nn", choices=["nn", "rf", "isolation_forest"], help="Model type")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (for NN)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (for NN)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (for NN)")
    parser.add_argument("--collision-weight", type=float, default=5.0, help="Weight for collision class")

    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data_dir,
        model_type=args.model_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        collision_weight=args.collision_weight,
    )

    # Load data
    samples = load_collision_data(config.data_dir)
    if not samples:
        logger.error(f"No data found in {config.data_dir}")
        return

    # Create feature extractor
    feature_extractor = FeatureExtractor(history_length=config.history_length)

    # Prepare training data
    features, labels, severity_labels, base_torques = prepare_training_data(
        samples, feature_extractor
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=config.test_split, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=config.val_split, random_state=42, stratify=y_train
    )

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Train model
    if config.model_type == "nn":
        train_dataset = CollisionDataset(X_train, y_train)
        val_dataset = CollisionDataset(X_val, y_val)

        model = train_neural_network(config, train_dataset, val_dataset, features.shape[1])
    elif config.model_type == "rf":
        model = train_random_forest(X_train, y_train)
    else:
        logger.info("Training Isolation Forest...")
        model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train)

    # Evaluate
    evaluate_model(model, X_test, y_test, config.model_type)

    # Save model
    save_model(model, args.output_dir, feature_extractor, base_torques, config.model_type)


if __name__ == "__main__":
    main()
