#!/usr/bin/env python3


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def split_csv(input_path: str,
              out_first: str = "First.csv",
              out_second: str = "Second.csv",
              frac: float = 0.6,
              random_state: int = 42) -> dict:
    
    df = pd.read_csv(input_path)
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * frac)
    first = df_shuffled.iloc[:split_idx]
    second = df_shuffled.iloc[split_idx:]
    first.to_csv(out_first, index=False)
    second.to_csv(out_second, index=False)
    return {
        "input_path": input_path,
        "out_first": out_first,
        "out_second": out_second,
        "total_rows": len(df),
        "first_rows": len(first),
        "second_rows": len(second)
    }

def label_encode_class(input_path: str,
                       column_name: str = None,
                       overwrite: bool = False,
                       out_path: str = None) -> dict:
    
    df = pd.read_csv(input_path)
    cols = df.columns.tolist()

    # Candidate target names (common)
    candidates = ["Class", "class", "Target", "target", "label", "Label", "y", "Y", "responded", "response"]
    chosen = column_name

    if not chosen:
        # direct match
        for c in candidates:
            if c in cols:
                chosen = c
                break
    if not chosen:
        # try case-insensitive match
        lower_map = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in lower_map:
                chosen = lower_map[cand.lower()]
                break
    if not chosen:
        # final fallback: last column (often the target)
        chosen = cols[-1]

    le = LabelEncoder()
    df["Label"] = le.fit_transform(df[chosen].astype(str))

    save_path = input_path if overwrite else (out_path or f"encoded_{os.path.basename(input_path)}")
    df.to_csv(save_path, index=False)

    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    return {
        "input_path": input_path,
        "used_column": chosen,
        "mapping": mapping,
        "saved_to": save_path,
        "shape": df.shape
    }

def evaluate_results(input_path: str,
                     positive_label = None,
                     roc_output: str = "roc_plot.png",
                     metrics_output: str = "results_metrics.txt") -> dict:
   
    df = pd.read_csv(input_path)
    required = ["ActualValues", "PredictedValues"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {input_path}. Found columns: {df.columns.tolist()}")

    # Encode string labels to integers consistently
    le = LabelEncoder()
    combined = pd.concat([df["ActualValues"].astype(str), df["PredictedValues"].astype(str)], axis=0)
    le.fit(combined)
    actual = le.transform(df["ActualValues"].astype(str))
    predicted = le.transform(df["PredictedValues"].astype(str))
    classes = list(le.classes_)

    if len(classes) < 2:
        raise ValueError("Need at least two distinct classes in ActualValues to compute metrics.")

    # choose positive label: if provided, map it; else choose encoded 1 if present, otherwise the larger encoded value
    if positive_label is not None:
        # allow passing either label string or encoded int
        if isinstance(positive_label, str):
            try:
                pos = int(le.transform([positive_label])[0])
            except Exception:
                raise ValueError(f"positive_label '{positive_label}' not found among classes: {classes}")
        else:
            pos = int(positive_label)
    else:
        # default heuristic
        encoded_values = sorted(set(actual.tolist() + predicted.tolist()))
        pos = 1 if 1 in encoded_values else max(encoded_values)

    # confusion matrix components for binary classification (treat pos as positive)
    TP = int(((predicted == pos) & (actual == pos)).sum())
    TN = int(((predicted != pos) & (actual != pos)).sum())
    FP = int(((predicted == pos) & (actual != pos)).sum())
    FN = int(((predicted != pos) & (actual == pos)).sum())
    total = len(df)

    accuracy = (TP + TN) / total if total > 0 else np.nan
    misclassification_rate = 1 - accuracy
    TPR = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    FPR = FP / (FP + TN) if (FP + TN) > 0 else np.nan
    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    counts = pd.Series(actual).value_counts()
    null_error_rate = counts.max() / total if total > 0 else np.nan

    
    try:
        pred_scores = pd.to_numeric(df["PredictedValues"], errors="coerce")
        if pred_scores.isnull().all():
            # fallback to using integer-encoded predicted labels as scores
            pred_scores = predicted
        else:
            # where numeric conversion produced NaN (e.g. label strings), fall back for those rows to encoded predicted
            pred_scores = pred_scores.fillna(pd.Series(predicted))
    except Exception:
        pred_scores = predicted

    fpr, tpr, thresholds = roc_curve(actual, pred_scores, pos_label=pos)
    roc_auc = auc(fpr, tpr)

    # Plot ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_output)
    plt.close()

    # Save metrics to text file
    with open(metrics_output, "w") as f:
        f.write(f"Classes (label encoder order): {classes}\n")
        f.write(f"Positive class encoded: {pos} -> '{le.inverse_transform([pos])[0]}'\n\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"TP: {TP}\nTN: {TN}\nFP: {FP}\nFN: {FN}\n\n")
        f.write(f"Accuracy: {accuracy:.6f}\n")
        f.write(f"Misclassification Rate: {misclassification_rate:.6f}\n")
        f.write(f"True Positive Rate (Recall / Sensitivity): {TPR:.6f}\n")
        f.write(f"False Positive Rate: {FPR:.6f}\n")
        f.write(f"Specificity: {specificity:.6f}\n")
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Null Error Rate: {null_error_rate:.6f}\n\n")
        f.write(f"ROC AUC: {roc_auc:.6f}\n\n")
        f.write("ROC points (fpr, tpr, threshold):\n")
        for x, y, th in zip(fpr, tpr, thresholds):
            f.write(f"{x:.6f}, {y:.6f}, threshold={th}\n")

    return {
        "input_path": input_path,
        "classes": classes,
        "positive_encoded": pos,
        "positive_label": le.inverse_transform([pos])[0],
        "total": total,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy": accuracy,
        "misclassification_rate": misclassification_rate,
        "TPR": TPR,
        "FPR": FPR,
        "specificity": specificity,
        "precision": precision,
        "null_error_rate": null_error_rate,
        "roc_auc": roc_auc,
        "roc_plot": roc_output,
        "metrics_file": metrics_output
    }


if __name__ == "__main__":
    # Example file paths (adjust if your files are in a different folder)
    house_file = "HouseData.csv"
    marketing_file = "Marketing.csv"
    results_file = "Results.csv"

    # 1) Split HouseData.csv (60/40)
    if os.path.exists(house_file):
        split_info = split_csv(house_file, out_first="First.csv", out_second="Second.csv", frac=0.6, random_state=42)
        print("Split done:", split_info)
    else:
        print("HouseData.csv not found at:", house_file)

    # 2) Label-encode Marketing.csv
    if os.path.exists(marketing_file):
        # If you know the exact column name, pass it as column_name="Class"
        label_info = label_encode_class(marketing_file, column_name=None, overwrite=False, out_path="Marketing_encoded.csv")
        print("Label encoding done:", label_info)
    else:
        print("Marketing.csv not found at:", marketing_file)

    # 3) Evaluate Results.csv
    if os.path.exists(results_file):
        metrics = evaluate_results(results_file, positive_label=None, roc_output="roc_plot.png", metrics_output="results_metrics.txt")
        print("Evaluation done. Metrics saved to:", metrics["metrics_file"])
        print("ROC plot at:", metrics["roc_plot"])
        print("Summary:", {k: metrics[k] for k in ["total", "TP", "TN", "FP", "FN", "accuracy", "roc_auc"]})
    else:
        print("Results.csv not found at:", results_file)
