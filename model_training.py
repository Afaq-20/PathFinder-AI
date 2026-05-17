"""
model_training.py

Train a RandomForestClassifier for AI career prediction using the
synthetic dataset `career_dataset.csv`.

The script performs:
- dataset loading and validation
- missing value handling
- feature/target separation
- label encoding
- numeric scaling
- train/test split
- model training and evaluation
- saving the trained model and label encoder
"""

import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = "career_dataset.csv"
MODEL_PATH = "career_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
TARGET_COLUMN = "Career Label"
FEATURE_COLUMNS = [
    "Coding_and_Algorithms",
    "UI_and_Visual_Design",
    "Data_and_Analytics",
    "Math_and_Predictive_Modeling",
    "Infrastructure_and_Automation",
    "Security_and_Networking",
    "Business_and_Product_Strategy",
    "System_Architecture_and_APIs",
]
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAMPLE_PREDICTIONS = 5


def load_dataset(path: str) -> pd.DataFrame:
    """Load the dataset from CSV and validate basic conditions."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    return df


def validate_columns(df: pd.DataFrame, required_columns) -> None:
    """Ensure the dataset contains all required feature and target columns."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and return a cleaned DataFrame."""
    df = df.copy()

    # Drop rows with missing target values
    df = df.dropna(subset=[TARGET_COLUMN])
    if df.empty:
        raise ValueError("Dataset has no rows after dropping missing target values.")

    # Fill missing feature values with column median
    feature_medians = df[FEATURE_COLUMNS].median()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(feature_medians)

    return df


def encode_target(series: pd.Series) -> tuple[LabelEncoder, np.ndarray]:
    """Encode career labels into integers."""
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(series)
    return encoder, encoded


def build_training_pipeline() -> Pipeline:
    """Build a training pipeline that scales numeric features and trains the model."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


def display_feature_importance(model: Pipeline, feature_names) -> None:
    """Print feature importances for the trained classifier."""
    if "classifier" not in model.named_steps:
        print("No classifier found in pipeline for feature importance.")
        return

    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        print("Classifier does not expose feature importance.")
        return

    importance_values = classifier.feature_importances_
    importance_pairs = sorted(
        zip(feature_names, importance_values), key=lambda x: x[1], reverse=True
    )

    print("\nFeature importance ranking:")
    for feature, importance in importance_pairs:
        print(f" - {feature}: {importance:.4f}")


def print_sample_predictions(
    model: Pipeline,
    encoder: LabelEncoder,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    sample_size: int = SAMPLE_PREDICTIONS,
) -> None:
    """Print a small set of test predictions alongside actual labels."""
    sample_size = min(sample_size, len(X_test))
    indices = random.sample(range(len(X_test)), sample_size)

    print(f"\nSample predictions for {sample_size} test rows:")
    for idx in indices:
        row = X_test.iloc[[idx]]
        prediction = model.predict(row)[0]
        actual = encoder.inverse_transform([y_test[idx]])[0]
        predicted_label = encoder.inverse_transform([prediction])[0]

        print(f"Row {idx}: predicted={predicted_label}, actual={actual}")
        print(row.to_dict(orient="records")[0])


def save_pickle(obj, path: str) -> None:
    """Persist an object to disk using pickle."""
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def main() -> None:
    print("Starting AI Career Prediction training...")

    df = load_dataset(DATA_PATH)
    validate_columns(df, FEATURE_COLUMNS + [TARGET_COLUMN])
    df = preprocess_data(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    encoder, y_encoded = encode_target(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    pipeline = build_training_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nModel accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=encoder.classes_))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    display_feature_importance(pipeline, FEATURE_COLUMNS)
    print_sample_predictions(pipeline, encoder, X_test, y_test)

    save_pickle(pipeline, MODEL_PATH)
    save_pickle(encoder, LABEL_ENCODER_PATH)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved label encoder to {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()
