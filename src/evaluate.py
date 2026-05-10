"""
evaluate.py - Comprehensive evaluation for Model A and Model B.
Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, Silhouette, R²
"""
import os, sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay, r2_score)

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import FeatureEngineering, load_race_dataset, expand_options, DEFAULT_RAW, DEFAULT_PROCESSED
from model_a_train  import (LogisticRegressionModel, SVMModel, KMeansClusteringModel,
                             EnsembleModel, build_feature_matrix, MODEL_A_DIR)
from model_b_train  import DistractorRanker, HintGenerator, MODEL_B_DIR

PROCESSED_DIR = DEFAULT_PROCESSED
RAW_DIR       = DEFAULT_RAW

def _compute_metrics(y_true, y_pred, name="Model"):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred).tolist()
    print(f"  [{name}] Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
    return {"accuracy":float(acc),"precision":float(prec),"recall":float(rec),
            "f1":float(f1),"confusion_matrix":cm,"model":name}

def _save_cm_plot(cm_list, title, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay(cm, display_labels=["Incorrect","Correct"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title); plt.tight_layout()
    plt.savefig(save_path, dpi=100); plt.close()

def plot_confusion_matrix_fig(cm_list, title="Confusion Matrix"):
    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay(cm, display_labels=["Incorrect","Correct"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title); plt.tight_layout()
    return fig

def plot_metrics_bar(metrics, title="Model Metrics"):
    keys   = ["accuracy","precision","recall","f1"]
    values = [metrics.get(k,0.0) for k in keys]
    labels = ["Accuracy","Precision","Recall","F1"]
    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(labels, values, color=["#4A90D9","#E87040","#50C878","#9B59B6"])
    ax.set_ylim(0,1.05); ax.set_title(title); ax.set_ylabel("Score")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout(); return fig

def evaluate_model_a_full(save_plots=True):
    print("="*55+"\nMODEL A FULL EVALUATION\n"+"="*55)
    results = {}
    try:
        fe  = FeatureEngineering.load(MODEL_A_DIR)
        lr  = LogisticRegressionModel.load(MODEL_A_DIR)
        svm = SVMModel.load(MODEL_A_DIR)
        ens = EnsembleModel.load(lr, svm, MODEL_A_DIR)
        km  = KMeansClusteringModel.load(MODEL_A_DIR)
    except Exception as e:
        print(f"  ⚠️  Could not load Model A: {e}"); return {}

    tp = os.path.join(PROCESSED_DIR,"test_expanded.csv")
    if os.path.exists(tp):
        test_exp = pd.read_csv(tp)
    else:
        _, _, test_raw = load_race_dataset(RAW_DIR)
        test_exp = expand_options(test_raw)

    X_test = build_feature_matrix(test_exp, fe)
    y_test  = test_exp["label"].values

    for name, model in [("LogisticRegression",lr),("SVM",svm),("Ensemble(LR+SVM)",ens)]:
        m = _compute_metrics(y_test, model.predict(X_test), name)
        results[name] = m
        if save_plots and "confusion_matrix" in m:
            _save_cm_plot(m["confusion_matrix"], f"CM — {name}",
                          os.path.join(PROCESSED_DIR, f"cm_{name.replace('(','').replace(')','').replace('+','_')}.png"))

    results["KMeans"] = {"silhouette_score": km.get_silhouette()}
    print(f"  KMeans Silhouette: {km.get_silhouette():.4f}")
    joblib.dump(results, os.path.join(MODEL_A_DIR,"eval_results.pkl"))
    print("Model A evaluation saved.")
    return results

def evaluate_model_b_full(save_plots=True):
    print("="*55+"\nMODEL B FULL EVALUATION\n"+"="*55)
    results = {}
    try:
        fe       = FeatureEngineering.load(MODEL_B_DIR)
        ranker   = DistractorRanker.load(MODEL_B_DIR)
        hint_gen = HintGenerator.load(MODEL_B_DIR)
    except Exception as e:
        print(f"  ⚠️  Could not load Model B: {e}"); return {}

    _, _, test_raw = load_race_dataset(RAW_DIR)
    print(f"  Test set size: {len(test_raw)} rows")

    X_d, y_d = ranker.build_training_data(test_raw, fe)   # full test set
    if len(y_d) > 1:
        m = _compute_metrics(y_d, ranker.model.predict(X_d), "DistractorRanker")
        results["DistractorRanker"] = m
        if save_plots and "confusion_matrix" in m:
            _save_cm_plot(m["confusion_matrix"],"CM — DistractorRanker",
                          os.path.join(PROCESSED_DIR,"cm_distractor_ranker.png"))

    X_h, y_h = hint_gen.build_training_data(test_raw)     # full test set
    if len(y_h) > 1:
        y_pred_h = hint_gen.model.predict(X_h)
        m = _compute_metrics(y_h, y_pred_h, "HintGenerator")
        proba = hint_gen.model.predict_proba(X_h)[:,1]
        m["r2_score"] = float(r2_score(y_h, proba))
        print(f"  HintGenerator R²={m['r2_score']:.4f}")
        results["HintGenerator"] = m

    joblib.dump(results, os.path.join(MODEL_B_DIR,"eval_results.pkl"))
    print("Model B evaluation saved.")
    return results

def load_saved_metrics():
    results = {}
    for key, dir_ in [("model_a",MODEL_A_DIR),("model_b",MODEL_B_DIR)]:
        for fname in ["eval_results.pkl","training_results.pkl"]:
            p = os.path.join(dir_, fname)
            if os.path.exists(p):
                results[key] = joblib.load(p); break
    return results

if __name__ == "__main__":
    evaluate_model_a_full()
    evaluate_model_b_full()
