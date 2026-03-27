import argparse
import os
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def load_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features file not found: {path}")
    df = pd.read_csv(path)
    return df


def build_feature_matrix(df: pd.DataFrame):
    features = [
        "tree_depth",
        "tree_size",
        "max_width",
        "avg_branching_factor",
        "leaf_count",
        "word_count",
        "sensational_word_count",
        "has_question_mark",
        "has_exclamation",
    ]
    X = df[features].fillna(0.0)
    y = df["label"].astype(int)
    return X, y, features


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight=None),
        "LinearSVC": LinearSVC(max_iter=5000, class_weight=None),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    }


def evaluate(clf, X_train, X_test, y_train, y_test, use_scaler=False):
    if use_scaler:
        clf.fit(X_train[0], y_train)
        y_pred = clf.predict(X_test[0])
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )
    report = classification_report(y_test, y_pred, target_names=["True", "False"], zero_division=0)
    return acc, prec, rec, f1, report, y_pred


def nice_print_results(name, acc, prec, rec, f1, report):
    print("\n" + "=" * 60)
    print(f"Model: {name}")
    print("-" * 60)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}   (for class=Fake)")
    print(f"Recall   : {rec:.4f}   (for class=Fake)")
    print(f"F1-score : {f1:.4f}   (for class=Fake)")
    print("-" * 60)
    print(report)


def main(args):
    df = load_data(args.input)
    X, y, feature_names = build_feature_matrix(df)

    # train/test split with stratify to keep class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Scale numeric features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models()

    results = []
    for name, clf in models.items():
        use_scaler = name in ["LogisticRegression", "LinearSVC"]

        # chọn ma trận training/test phù hợp (2D)
        if use_scaler:
            Xtr, Xte = X_train_scaled, X_test_scaled
        else:
            Xtr, Xte = X_train.values, X_test.values

        # fit & predict
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)

        # metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", pos_label=1, zero_division=0
        )
        report = classification_report(y_test, y_pred, target_names=["True", "False"], zero_division=0)

        nice_print_results(name, acc, prec, rec, f1, report)

        results.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

        if args.save_models:
            os.makedirs(args.model_dir, exist_ok=True)
            joblib.dump(clf, os.path.join(args.model_dir, f"{name}.pkl"))

    # # summary
    # print("\n" + "#" * 60)
    # print("SUMMARY (table)")
    # print("#" * 60)
    # print("Model,Accuracy,Precision(Fake),Recall(Fake),F1(Fake)")
    # for r in results:
    #     print(f"{r['model']},{r['accuracy']:.4f},{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f}")
    
    # summary table
    print("\n" + "#" * 60)
    print(f"{'SUMMARY TABLE':^60}")
    print("#" * 60)
    print(f"{'Model':<20} | {'Acc':<8} | {'Prec(F)':<8} | {'Rec(F)':<8} | {'F1(F)':<8}")
    print("-" * 60)
    
    best = None
    for r in results:
        print(f"{r['model']:<20} | {r['accuracy']:<8.4f} | {r['precision']:<8.4f} | {r['recall']:<8.4f} | {r['f1']:<8.4f}")
        if best is None or r['accuracy'] > best['accuracy']:
            best = r

    # Final accuracy / best model
    if best:
        print("\n" + "=" * 60)
        print(f" WINNING MODEL: {best['model']}")
        print(f" Final Accuracy: {best['accuracy']:.4f}")
        print(f" Final F1 (Fake): {best['f1']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline classifiers on extracted_features.csv")
    parser.add_argument("--input", default="data/processed/extracted_features.csv", help="Path to extracted features CSV")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--save-models", action="store_true", help="Save trained models")
    parser.add_argument("--model-dir", default="models", help="Directory to save models")
    args = parser.parse_args()

    main(args)

