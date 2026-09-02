"""CLI entry point for the diabetes prediction project.

Usage:
    python main.py train
    python main.py predict --pregnancies 2 --glucose 130 --blood-pressure 70 \
        --skin-thickness 25 --insulin 100 --bmi 28.5 --dpf 0.45 --age 33
"""

import argparse
import sys

from src.data.load_data import load_config, load_data
from src.data.preprocess import clean_data
from src.features.feature_engineering import (
    split_features_target,
    split_train_test,
    scale_features,
)
from src.models.train import train_logistic_regression, save_artifacts
from src.models.evaluate import evaluate_model
from src.models.predict import load_artifacts, predict_one, FEATURE_ORDER


def run_train():
    config = load_config()
    df = load_data(config)
    df_clean = clean_data(df, config["data"]["invalid_zero_columns"])

    X, y = split_features_target(df_clean, config["data"]["target_column"])
    X_train, X_test, y_train, y_test = split_train_test(X, y, config)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model = train_logistic_regression(X_train_scaled, y_train, config)
    results = evaluate_model(model, X_test_scaled, y_test)

    print("Training complete.")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"ROC-AUC:  {results['roc_auc']:.4f}")
    print()
    print("Classification report:")
    print(results["classification_report"])

    save_artifacts(model, scaler)
    print("Model and scaler saved to models/")


def run_predict(args):
    features = {
        "Pregnancies": args.pregnancies,
        "Glucose": args.glucose,
        "BloodPressure": args.blood_pressure,
        "SkinThickness": args.skin_thickness,
        "Insulin": args.insulin,
        "BMI": args.bmi,
        "DiabetesPedigreeFunction": args.dpf,
        "Age": args.age,
    }

    model, scaler = load_artifacts()
    result = predict_one(features, model, scaler)

    label = "Diabetic" if result["prediction"] == 1 else "Non-diabetic"
    print(f"Prediction: {label} (class {result['prediction']})")
    print(f"Probability of diabetes: {result['probability']:.4f}")


def build_parser():
    parser = argparse.ArgumentParser(description="Diabetes risk prediction pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train the model and save artifacts.")

    predict_parser = subparsers.add_parser("predict", help="Predict for a single patient.")
    predict_parser.add_argument("--pregnancies", type=int, required=True)
    predict_parser.add_argument("--glucose", type=float, required=True)
    predict_parser.add_argument("--blood-pressure", type=float, required=True)
    predict_parser.add_argument("--skin-thickness", type=float, required=True)
    predict_parser.add_argument("--insulin", type=float, required=True)
    predict_parser.add_argument("--bmi", type=float, required=True)
    predict_parser.add_argument("--dpf", type=float, required=True, help="DiabetesPedigreeFunction")
    predict_parser.add_argument("--age", type=int, required=True)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train()
    elif args.command == "predict":
        run_predict(args)


if __name__ == "__main__":
    sys.exit(main())
