import os
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, jaccard_score, balanced_accuracy_score,
    matthews_corrcoef, fbeta_score
)

def load_binary_mask(path, threshold=True):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not open or find the image: {path}")
    if threshold:
        # Standardize to 0 and 1
        mask = (mask > 127).astype(np.uint8) 
    return mask

def evaluate_binary_metrics(gt, pred):
    # Flatten to 1D arrays for sklearn metrics
    gt = gt.flatten()
    pred = pred.flatten()

    # Manual calculation for Specificity and smoothing
    tp = np.sum((gt == 1) & (pred == 1))
    tn = np.sum((gt == 0) & (pred == 0))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    smooth = 1e-6

    metrics = {
        "Accuracy": accuracy_score(gt, pred),
        "Precision": precision_score(gt, pred, zero_division=0),
        "Recall (Sensitivity)": recall_score(gt, pred, zero_division=0),
        "Specificity": tn / (tn + fp + smooth),
        "Dice (F1)": f1_score(gt, pred, zero_division=0),
        "IoU (Jaccard)": jaccard_score(gt, pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(gt, pred),
        "F2 Score": fbeta_score(gt, pred, beta=2, zero_division=0),
        "MCC": matthews_corrcoef(gt, pred),
    }

    return metrics

def evaluate_single_map(gt_path, pred_path, output_csv="map_eval_results.csv"):
    print(f"--- Loading Maps ---")
    gt = load_binary_mask(gt_path)
    pred = load_binary_mask(pred_path)

    if gt.shape != pred.shape:
        print(f"⚠️ Shape Mismatch! GT: {gt.shape}, Pred: {pred.shape}")
        print("Attempting to resize prediction to match Ground Truth...")
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    print(f"Calculating metrics for {gt.size:,} pixels...")
    results = evaluate_binary_metrics(gt, pred)
    
    # Convert to DataFrame for easy viewing/saving
    df = pd.DataFrame([results])
    df.to_csv(output_csv, index=False)
    
    print("\n--- Results ---")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
        
    return df

if __name__ == "__main__":
    # Update these to the direct paths of your two full maps
    gt_map_path = 'data/Output/Melbourne/Truths/Melbourne/cleaned/Melbourne_Truth.png'
    pred_map_path = 'data/Output/Melbourne_4-03/Vectorisation/cleaned/vc000868-001.png'
    evaluate_single_map(gt_map_path, pred_map_path)