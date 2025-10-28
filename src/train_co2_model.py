"""Train CO₂ emission predictors with classical and deep learning models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NON_SEQUENCE_MODELS = {"dt", "rf", "xgb", "svm", "ann"}
SEQUENCE_MODELS = {"rnn", "lstm"}


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


def expand_paths(paths: Sequence[str]) -> List[Path]:
    """Expand glob patterns and validate that files exist."""

    expanded: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if any(ch in raw for ch in "*?[]"):
            expanded.extend(Path().glob(raw))
        elif path.exists():
            expanded.append(path)
        else:
            raise FileNotFoundError(f"Input path does not exist: {raw}")

    if not expanded:
        raise FileNotFoundError("No training files matched the supplied paths")
    return sorted(expanded)


def load_table(path: Path, sheet: Optional[str], exclude_sheets: Iterable[str]) -> pd.DataFrame:
    """Read a CSV or Excel table and return a dataframe."""

    exclude = {s.strip().lower() for s in exclude_sheets if s}
    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx", ".xlsm"}:
        excel = pd.ExcelFile(path)
        sheet_names = [sheet] if sheet else excel.sheet_names
        frames = []
        for sh in sheet_names:
            if sh.strip().lower() in exclude:
                continue
            try:
                frame = pd.read_excel(excel, sheet_name=sh)
            except ValueError:
                continue
            frames.append(frame)
        if not frames:
            raise ValueError(f"No usable sheets found in {path}")
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path)

    df = df.dropna(how="all").fillna(0)
    return df


def load_datasets(
    paths: Sequence[str],
    sheet: Optional[str],
    exclude_sheets: Iterable[str],
) -> pd.DataFrame:
    frames = [load_table(path, sheet, exclude_sheets) for path in expand_paths(paths)]
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise ValueError("Training dataframe is empty after loading inputs")
    return combined


def detect_target_column(df: pd.DataFrame, explicit: Optional[str]) -> str:
    if explicit:
        if explicit not in df.columns:
            raise KeyError(f"Target column '{explicit}' not found in dataframe")
        return explicit
    for col in df.columns:
        if "co2" in str(col).lower():
            return col
    raise ValueError("Unable to detect target column. Please specify --target-column explicitly.")


def determine_features(
    df: pd.DataFrame,
    target: str,
    feature_columns: Optional[Sequence[str]],
) -> List[str]:
    if feature_columns:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Feature columns missing from dataframe: {missing}")
        return list(feature_columns)
    numeric_cols = [c for c in df.columns if c != target and np.issubdtype(df[c].dtype, np.number)]
    if not numeric_cols:
        raise ValueError("No numeric feature columns available for training")
    return numeric_cols


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def build_non_sequence_pipeline(model_name: str, features: Sequence[str]) -> Pipeline:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    if model_name == "dt":
        estimator = DecisionTreeRegressor(random_state=42)
    elif model_name == "rf":
        estimator = RandomForestRegressor(n_estimators=400, random_state=42)
    elif model_name == "svm":
        estimator = SVR(kernel="rbf", C=10.0, gamma="scale")
    elif model_name == "ann":
        estimator = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            learning_rate_init=1e-3,
            max_iter=500,
            random_state=42,
        )
    elif model_name == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for model='xgb'. Install it with `pip install xgboost`."
            ) from exc
        estimator = XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model '{model_name}'")

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), list(features))]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def build_sequence_model(model_name: str, input_shape: Tuple[int, int]):
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for recurrent models. Install it with `pip install tensorflow`."
        ) from exc

    units = 128
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    if model_name == "rnn":
        model.add(keras.layers.SimpleRNN(units, return_sequences=False))
    elif model_name == "lstm":
        model.add(keras.layers.LSTM(units, return_sequences=False))
    else:
        raise ValueError(f"Unsupported sequence model '{model_name}'")
    model.add(keras.layers.Dense(units // 2, activation="relu"))
    model.add(keras.layers.Dense(1, activation="linear"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def prepare_sequence_data(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if sequence_length < 1:
        raise ValueError("--sequence-length must be >= 1")

    values = df[features + [target]].to_numpy(dtype=np.float32)
    X, y = [], []
    for idx in range(len(values) - sequence_length):
        window = values[idx : idx + sequence_length]
        X.append(window[:, :-1])
        y.append(values[idx + sequence_length, -1])
    if not X:
        raise ValueError("Not enough rows to create sequences. Reduce --sequence-length or collect more data.")
    return np.asarray(X), np.asarray(y)


def train_non_sequence_model(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    model_name: str,
    test_size: float,
    random_state: int,
) -> Tuple[Pipeline, Dict[str, float], np.ndarray, np.ndarray]:
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipeline = build_non_sequence_pipeline(model_name, features)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test, predictions)
    return pipeline, metrics, y_test.to_numpy(dtype=np.float32), predictions.astype(np.float32)


def train_sequence_model(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    model_name: str,
    sequence_length: int,
    test_size: float,
    epochs: int,
    batch_size: int,
) -> Tuple[object, Dict[str, float], np.ndarray, np.ndarray, StandardScaler]:
    X, y = prepare_sequence_data(df, features, target, sequence_length)

    split_idx = int(len(X) * (1 - test_size))
    if split_idx == 0 or split_idx == len(X):
        raise ValueError("test_size produces an empty train/test split for the sequence data")

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    model = build_sequence_model(model_name, (sequence_length, len(features)))
    model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    predictions = model.predict(X_test_scaled).reshape(-1)
    metrics = evaluate_predictions(y_test, predictions)
    return model, metrics, y_test, predictions, scaler


def make_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, y_true, label="Experiment", color="black", linewidth=1.0)
    ax.plot(xs, y_pred, label="Predicted", color="red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("CO₂ concentration")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_series.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.6, s=14)
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Predicted")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_scatter.png", dpi=250)
    plt.close(fig)


def evaluate_on_validation(
    df_val: pd.DataFrame,
    features: Sequence[str],
    target: str,
    model_type: str,
    fitted_model,
    scaler: Optional[StandardScaler],
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if model_type in NON_SEQUENCE_MODELS:
        X_val = df_val[features]
        y_val = df_val[target].to_numpy(dtype=np.float32)
        predictions = fitted_model.predict(X_val)
    else:
        X_val, y_val = prepare_sequence_data(df_val, features, target, sequence_length)
        if scaler is None:
            raise RuntimeError("Scaler is required for sequence validation evaluation")
        X_val_flat = X_val.reshape(len(X_val), -1)
        X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
        predictions = fitted_model.predict(X_val_scaled).reshape(-1)
    metrics = evaluate_predictions(y_val, predictions)
    return y_val, predictions, metrics


def save_model(
    model,
    model_type: str,
    output_path: Path,
    scaler: Optional[StandardScaler],
) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model_type in NON_SEQUENCE_MODELS:
        joblib.dump(model, output_path)
    else:
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise ImportError("TensorFlow must be installed to save sequence models") from exc
        if output_path.suffix:
            raise ValueError(
                "For sequence models please provide --model-output as a directory path (without file extension)."
            )
        output_path.mkdir(parents=True, exist_ok=True)
        keras_path = output_path / "model.keras"
        model.save(keras_path)
        if scaler is not None:
            joblib.dump(scaler, output_path / "scaler.joblib")


# ---------------------------------------------------------------------------
# Argument parsing and main entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "train",
        nargs="+",
        help="Training datasets (CSV/Excel). Globs such as 'data/*.csv' are supported.",
    )
    parser.add_argument(
        "--sheet",
        help="Sheet name to read from Excel files. If omitted, all sheets are used.",
    )
    parser.add_argument(
        "--exclude-sheet",
        action="append",
        default=[],
        help="Sheet names to exclude when reading Excel workbooks (can be repeated).",
    )
    parser.add_argument(
        "--target-column",
        help="Target column to predict. Defaults to the first column containing 'co2'.",
    )
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        help="Explicit feature columns to use. Defaults to all numeric columns except the target.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(NON_SEQUENCE_MODELS | SEQUENCE_MODELS),
        default="rf",
        help="Which model to train (default: rf).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the data reserved for evaluation (between 0 and 1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for train/test splitting of non-sequence models.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
        help="Sliding window size for sequence models (ignored for non-sequence models).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for sequence models (ignored for other models).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for sequence models (ignored for other models).",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("runs/co2_model.joblib"),
        help="Where to save the trained model. For sequence models provide a directory path.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optional JSON file to store evaluation metrics.",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        help="Optional validation dataset (CSV/Excel) for an additional evaluation pass.",
    )
    parser.add_argument(
        "--validation-sheet",
        help="Sheet name for the validation workbook (if applicable).",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        help="If supplied, save time-series and scatter plots to this directory.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    df_train = load_datasets(args.train, args.sheet, args.exclude_sheet)
    target = detect_target_column(df_train, args.target_column)
    features = determine_features(df_train, target, args.feature_columns)

    if args.model in NON_SEQUENCE_MODELS:
        model, metrics, y_true, y_pred = train_non_sequence_model(
            df_train, features, target, args.model, args.test_size, args.random_state
        )
        scaler = None
    else:
        model, metrics, y_true, y_pred, scaler = train_sequence_model(
            df_train,
            features,
            target,
            args.model,
            args.sequence_length,
            args.test_size,
            args.epochs,
            args.batch_size,
        )

    print("Evaluation metrics on hold-out set:")
    for key, value in metrics.items():
        print(f"  {key.upper()}: {value:.4f}")

    if args.plots_dir:
        make_plots(y_true, y_pred, args.plots_dir, prefix="holdout")

    if args.validation:
        df_val = load_table(args.validation, args.validation_sheet, [])
        y_val, y_val_pred, val_metrics = evaluate_on_validation(
            df_val, features, target, args.model, model, scaler, args.sequence_length
        )
        print("Validation metrics:")
        for key, value in val_metrics.items():
            print(f"  {key.upper()}: {value:.4f}")
        if args.plots_dir:
            make_plots(y_val, y_val_pred, args.plots_dir, prefix="validation")

    if args.metrics_output:
        payload = {"holdout": metrics}
        if args.validation:
            payload["validation"] = val_metrics
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_output.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    save_model(model, args.model, args.model_output, scaler)
    print(f"Model saved to {args.model_output}")


if __name__ == "__main__":
    main()
