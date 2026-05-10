"""
model_a_train.py - Model A: Question & Answer Generator / Verifier
Implements: Logistic Regression, SVM, K-Means Clustering, Template QGen, Soft-Vote Ensemble
"""
import os, sys, numpy as np, pandas as pd, joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import (FeatureEngineering, load_race_dataset, expand_options,
                            run_preprocessing_pipeline, split_sentences,
                            sentence_keyword_overlap, tokenize, remove_stopwords, clean_text,
                            BASE_DIR, DEFAULT_RAW, DEFAULT_PROCESSED)

MODEL_A_DIR = os.path.join(BASE_DIR, "models", "model_a", "traditional")
os.makedirs(MODEL_A_DIR, exist_ok=True)

WH_MAP = {
    "who":   ("person","people","scientist","author","president","leader"),
    "where": ("place","city","country","located","region","area"),
    "when":  ("year","century","date","time","period","age"),
    "why":   ("reason","because","therefore","cause","purpose"),
    "how":   ("number","amount","many","much","rate","percent","process"),
}

def _pick_wh(sentence):
    s = sentence.lower()
    for wh, keywords in WH_MAP.items():
        if any(k in s for k in keywords): return wh
    return "what"

def generate_questions_from_passage(passage, answer=""):
    sents = split_sentences(passage)
    candidates = []
    for sent in sents:
        if len(sent.split()) < 5: continue
        score = sentence_keyword_overlap(sent, answer) if answer else 0.4
        wh = _pick_wh(sent)
        tokens = remove_stopwords(sent.split())
        rest = " ".join(tokens[:min(6, len(tokens))])
        q = f"{wh.capitalize()} is {rest}?"
        candidates.append((q, sent, score))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates

def select_best_question(candidates, fe=None):
    if not candidates:
        return ("What is the main idea of this passage?", "", 0.5)
    return candidates[0]

def build_feature_matrix(df_exp, fe):
    art = df_exp["article"].fillna("").astype(str)
    qst = df_exp["question"].fillna("").astype(str)
    opt = df_exp["option"].fillna("").astype(str)
    combined = (art + " " + qst + " " + opt).tolist()
    X_ohe = fe.transform_ohe(combined)
    X_lex = csr_matrix(fe.lexical_features(df_exp))
    X_cos = csr_matrix(fe.cosine_features(df_exp))
    return hstack([X_ohe, X_lex, X_cos])

def cosine_similarity_report(df_exp, fe, label="Dataset", sample=5000):
    """Print mean cosine similarity for correct vs incorrect options.
    Samples up to `sample` rows to keep it fast on large datasets.
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    import numpy as np

    df = df_exp.copy()
    if len(df) > sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)

    art = df["article"].fillna("").astype(str).tolist()
    qst = df["question"].fillna("").astype(str).tolist()
    opt = df["option"].fillna("").astype(str).tolist()
    labels = df["label"].values

    A = fe.transform_ohe(art)   # article vectors
    Q = fe.transform_ohe(qst)   # question vectors
    O = fe.transform_ohe(opt)   # option vectors

    ao_sims = np.array([float(cos_sim(A[i], O[i])[0][0]) for i in range(len(df))])
    qo_sims = np.array([float(cos_sim(Q[i], O[i])[0][0]) for i in range(len(df))])

    correct   = labels == 1
    incorrect = labels == 0

    print(f"\n  ╔══ Cosine Similarity Report — {label} {'(sampled)' if len(df_exp) > sample else ''} ══")
    print(f"  ║  Article  ↔  Correct  Option  : {ao_sims[correct].mean():.4f}")
    print(f"  ║  Article  ↔  Incorrect Option : {ao_sims[incorrect].mean():.4f}")
    print(f"  ║  Question ↔  Correct  Option  : {qo_sims[correct].mean():.4f}")
    print(f"  ║  Question ↔  Incorrect Option : {qo_sims[incorrect].mean():.4f}")
    gap_ao = ao_sims[correct].mean() - ao_sims[incorrect].mean()
    gap_qo = qo_sims[correct].mean() - qo_sims[incorrect].mean()
    print(f"  ║  Gap (Article↔Option)          : {gap_ao:+.4f}  {'✓ discriminative' if gap_ao > 0 else '✗ low signal'}")
    print(f"  ║  Gap (Question↔Option)         : {gap_qo:+.4f}  {'✓ discriminative' if gap_qo > 0 else '✗ low signal'}")
    print(f"  ╚{'═'*50}")

    return {"ao_correct": float(ao_sims[correct].mean()),
            "ao_incorrect": float(ao_sims[incorrect].mean()),
            "qo_correct": float(qo_sims[correct].mean()),
            "qo_incorrect": float(qo_sims[incorrect].mean()),
            "ao_gap": float(gap_ao), "qo_gap": float(gap_qo)}

def evaluate_model(model, X, y_true, name="Model"):
    y_pred = model.predict(X)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n  ┌─── {name} ───")
    print(f"  │  Binary Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  │  Macro Precision   : {prec:.4f}")
    print(f"  │  Macro Recall      : {rec:.4f}")
    print(f"  │  Macro F1-Score    : {f1:.4f}")
    print(f"  └──────────────────────────────")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "confusion_matrix": cm.tolist(), "model": name}

# ── Logistic Regression ──────────────────────────────────────────────────────
class LogisticRegressionModel:
    def __init__(self, C=1.0):
        self.model = LogisticRegression(C=C, max_iter=1000, solver="saga", n_jobs=-1, random_state=42)
        self.name = "LogisticRegression"
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def predict_proba(self, X): return self.model.predict_proba(X)
    def save(self, path=MODEL_A_DIR):
        joblib.dump(self.model, os.path.join(path,"logistic_regression.pkl"))
    @staticmethod
    def load(path=MODEL_A_DIR):
        obj = LogisticRegressionModel()
        obj.model = joblib.load(os.path.join(path,"logistic_regression.pkl"))
        return obj

# ── SVM ─────────────────────────────────────────────────────────────────────
class SVMModel:
    def __init__(self, C=1.0):
        self.model = CalibratedClassifierCV(LinearSVC(C=C, max_iter=2000, random_state=42), cv=3)
        self.name = "SVM"
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def predict_proba(self, X): return self.model.predict_proba(X)
    def save(self, path=MODEL_A_DIR):
        joblib.dump(self.model, os.path.join(path,"svm.pkl"))
    @staticmethod
    def load(path=MODEL_A_DIR):
        obj = SVMModel()
        obj.model = joblib.load(os.path.join(path,"svm.pkl"))
        return obj

# ── K-Means ──────────────────────────────────────────────────────────────────
class KMeansClusteringModel:
    def __init__(self, n_clusters=4, n_components=50):
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.silhouette = None; self.name = "KMeans"
    def fit(self, X, sample_size=3000):
        idx = np.random.choice(X.shape[0], min(sample_size, X.shape[0]), replace=False)
        Xs = X[idx]
        Xr = self.svd.fit_transform(Xs)
        self.kmeans.fit(Xr)
        labels = self.kmeans.labels_
        self.silhouette = silhouette_score(Xr, labels, sample_size=min(1000, len(labels))) if len(set(labels))>1 else 0.0
        print(f"  KMeans Silhouette: {self.silhouette:.4f}")
        return self
    def predict(self, X): return self.kmeans.predict(self.svd.transform(X))
    def get_silhouette(self): return self.silhouette
    def save(self, path=MODEL_A_DIR):
        joblib.dump({"kmeans":self.kmeans,"svd":self.svd,"silhouette":self.silhouette}, os.path.join(path,"kmeans.pkl"))
    @staticmethod
    def load(path=MODEL_A_DIR):
        obj = KMeansClusteringModel()
        d = joblib.load(os.path.join(path,"kmeans.pkl"))
        obj.kmeans=d["kmeans"]; obj.svd=d["svd"]; obj.silhouette=d["silhouette"]
        return obj

# ── Ensemble ─────────────────────────────────────────────────────────────────
class EnsembleModel:
    def __init__(self, lr, svm, lw=0.5, sw=0.5):
        self.lr=lr; self.svm=svm; self.lw=lw; self.sw=sw; self.name="Ensemble(LR+SVM)"
    def predict_proba(self, X): return self.lw*self.lr.predict_proba(X) + self.sw*self.svm.predict_proba(X)
    def predict(self, X): return np.argmax(self.predict_proba(X), axis=1)
    def save(self, path=MODEL_A_DIR):
        joblib.dump({"lw":self.lw,"sw":self.sw}, os.path.join(path,"ensemble_weights.pkl"))
    @staticmethod
    def load(lr, svm, path=MODEL_A_DIR):
        fp = os.path.join(path,"ensemble_weights.pkl")
        if os.path.exists(fp):
            d = joblib.load(fp); return EnsembleModel(lr, svm, d["lw"], d["sw"])
        return EnsembleModel(lr, svm)

def train_model_a(data_dir=None, processed_dir=None):
    if data_dir is None: data_dir = DEFAULT_RAW
    if processed_dir is None: processed_dir = DEFAULT_PROCESSED
    print("="*55+"\nMODEL A TRAINING\n"+"="*55)
    tp = os.path.join(processed_dir,"train_expanded.csv")
    vp = os.path.join(processed_dir,"val_expanded.csv")
    if os.path.exists(tp):
        train_exp = pd.read_csv(tp); val_exp = pd.read_csv(vp)
        fe = FeatureEngineering.load(MODEL_A_DIR)
    else:
        train_exp, val_exp, _, fe = run_preprocessing_pipeline(data_dir, processed_dir)

    print("\n[1] Building feature matrices...")
    X_tr = build_feature_matrix(train_exp, fe); y_tr = train_exp["label"].values
    X_val= build_feature_matrix(val_exp, fe);   y_val= val_exp["label"].values
    print(f"  X_train: {X_tr.shape}  X_val: {X_val.shape}")

    print("\n[2] Cosine Similarity Analysis...")
    cos_train = cosine_similarity_report(train_exp, fe, label="Train")
    cos_val   = cosine_similarity_report(val_exp,   fe, label="Val")

    results = {"cosine_similarity": {"train": cos_train, "val": cos_val}}
    print("\n[3] Training Logistic Regression...")
    lr = LogisticRegressionModel(); lr.fit(X_tr, y_tr); lr.save()
    results["LogisticRegression"] = evaluate_model(lr, X_val, y_val, "LogisticRegression")

    print("\n[4] Training SVM...")
    svm = SVMModel(); svm.fit(X_tr, y_tr); svm.save()
    results["SVM"] = evaluate_model(svm, X_val, y_val, "SVM")

    print("\n[5] Training K-Means...")
    km = KMeansClusteringModel(); km.fit(X_tr); km.save()
    results["KMeans"] = {"silhouette_score": km.get_silhouette()}

    print("\n[6] Ensemble (Soft Voting)...")
    ens = EnsembleModel(lr, svm); ens.save()
    results["Ensemble(LR+SVM)"] = evaluate_model(ens, X_val, y_val, "Ensemble(LR+SVM)")

    joblib.dump(results, os.path.join(MODEL_A_DIR,"training_results.pkl"))
    print("\n✅ Model A training complete!")
    return results, lr, svm, km, ens, fe

if __name__ == "__main__":
    train_model_a(DEFAULT_RAW, DEFAULT_PROCESSED)
