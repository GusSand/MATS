"""
Probe Trainer for SR/SCG Separation Experiment

Trains linear probes and computes direction similarity.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Dict, List, Tuple, Optional


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


class ProbeTrainer:
    def __init__(self):
        self.probes = {}
        self.scalers = {}
        self.directions = {}

    def train_probe_at_layer(self, X: np.ndarray, y: np.ndarray, layer_idx: int,
                             probe_name: str = "probe") -> dict:
        """Train a logistic regression probe at one layer and extract direction."""
        key = f"{probe_name}_layer_{layer_idx}"

        if len(X) < 10:
            return {
                'layer': layer_idx,
                'probe_name': probe_name,
                'n_samples': len(X),
                'accuracy': None,
                'error': 'Not enough samples'
            }

        # Check class balance
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            return {
                'layer': layer_idx,
                'probe_name': probe_name,
                'n_samples': len(X),
                'accuracy': None,
                'error': f'Only one class present: {unique[0]}'
            }

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError as e:
            # Not enough samples for stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=42)

        try:
            # Cross-validation on training set
            cv_scores = cross_val_score(clf, X_train, y_train, cv=min(5, len(y_train) // 2))

            # Final fit and test
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5

            # Store probe and scaler
            self.probes[key] = clf
            self.scalers[key] = scaler

            # Extract direction (normalized coefficient vector)
            direction = clf.coef_[0]
            direction_normalized = direction / np.linalg.norm(direction)
            self.directions[key] = direction_normalized

            return {
                'layer': layer_idx,
                'probe_name': probe_name,
                'n_samples': len(X),
                'n_train': len(X_train),
                'n_test': len(X_test),
                'accuracy': float(accuracy),
                'cv_accuracy_mean': float(cv_scores.mean()),
                'cv_accuracy_std': float(cv_scores.std()),
                'auc': float(auc),
                'class_balance': float(y.mean()),
                'direction_norm': float(np.linalg.norm(direction))
            }
        except Exception as e:
            return {
                'layer': layer_idx,
                'probe_name': probe_name,
                'n_samples': len(X),
                'accuracy': None,
                'error': str(e)
            }

    def train_all_layers(self, data: dict, probe_name: str) -> List[dict]:
        """Train probes at all layers."""
        results = []
        n_layers = len(data)

        for layer_idx in range(n_layers):
            X = data[layer_idx]['X']
            y = data[layer_idx]['y']
            result = self.train_probe_at_layer(X, y, layer_idx, probe_name)
            results.append(result)

        return results

    def get_direction(self, probe_name: str, layer_idx: int) -> Optional[np.ndarray]:
        """Get the normalized direction vector for a probe at a layer."""
        key = f"{probe_name}_layer_{layer_idx}"
        return self.directions.get(key)

    def compute_direction_similarity(self, probe1_name: str, probe2_name: str,
                                     layer_idx: int) -> Optional[float]:
        """Compute cosine similarity between two probe directions at a layer."""
        dir1 = self.get_direction(probe1_name, layer_idx)
        dir2 = self.get_direction(probe2_name, layer_idx)

        if dir1 is None or dir2 is None:
            return None

        return compute_cosine_similarity(dir1, dir2)

    def compute_all_similarities(self, probe1_name: str, probe2_name: str,
                                 n_layers: int) -> List[dict]:
        """Compute direction similarities across all layers."""
        results = []

        for layer_idx in range(n_layers):
            similarity = self.compute_direction_similarity(probe1_name, probe2_name, layer_idx)
            results.append({
                'layer': layer_idx,
                'cosine_similarity': similarity
            })

        return results

    def predict_with_probe(self, X: np.ndarray, probe_name: str, layer_idx: int) -> dict:
        """Make predictions using a trained probe."""
        key = f"{probe_name}_layer_{layer_idx}"

        if key not in self.probes:
            return {'error': f'Probe {key} not found'}

        clf = self.probes[key]
        scaler = self.scalers[key]

        X_scaled = scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)
        pred = clf.predict(X_scaled)
        prob = clf.predict_proba(X_scaled)[:, 1]

        return {
            'prediction': int(pred[0]) if len(pred) == 1 else pred.tolist(),
            'probability': float(prob[0]) if len(prob) == 1 else prob.tolist()
        }

    def compute_mean_direction(self, X_class1: np.ndarray, X_class0: np.ndarray) -> np.ndarray:
        """
        Compute direction as difference of class means.
        Alternative to probe coefficients for direction extraction.

        direction = mean(class1) - mean(class0)
        """
        mean1 = np.mean(X_class1, axis=0)
        mean0 = np.mean(X_class0, axis=0)
        direction = mean1 - mean0
        return direction / np.linalg.norm(direction)


def extract_directions_from_data(data: dict, n_layers: int) -> dict:
    """
    Extract direction vectors from activation data using mean difference.

    This gives the "raw" direction without probe training.
    """
    directions = {}

    for layer_idx in range(n_layers):
        X = data[layer_idx]['X']
        y = data[layer_idx]['y']

        if len(X) == 0:
            directions[layer_idx] = None
            continue

        X_class1 = X[y == 1]
        X_class0 = X[y == 0]

        if len(X_class1) == 0 or len(X_class0) == 0:
            directions[layer_idx] = None
            continue

        mean1 = np.mean(X_class1, axis=0)
        mean0 = np.mean(X_class0, axis=0)
        direction = mean1 - mean0

        # Store both normalized and raw
        directions[layer_idx] = {
            'raw': direction,
            'normalized': direction / np.linalg.norm(direction),
            'norm': np.linalg.norm(direction)
        }

    return directions
