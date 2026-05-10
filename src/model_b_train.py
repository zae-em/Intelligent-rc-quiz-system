"""
model_b_train.py - Model B: Distractor & Hint Generator
Implements: Candidate extraction, OHE cosine similarity ranking, LR-based distractor ranker, extractive hint generator
"""
import os, sys, numpy as np, pandas as pd, joblib
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import (FeatureEngineering, load_race_dataset, expand_options,
                            run_preprocessing_pipeline, split_sentences,
                            sentence_keyword_overlap, tokenize, remove_stopwords, clean_text,
                            BASE_DIR, DEFAULT_RAW, DEFAULT_PROCESSED)

MODEL_B_DIR = os.path.join(BASE_DIR, "models", "model_b", "traditional")
os.makedirs(MODEL_B_DIR, exist_ok=True)

# ── Candidate Extraction ─────────────────────────────────────────────────────
def extract_candidate_words(passage, correct_answer, top_n=20):
    tokens = remove_stopwords(tokenize(passage))
    ans_tokens = set(tokenize(correct_answer))
    filtered = [t for t in tokens if t not in ans_tokens and len(t) > 2]
    freq = Counter(filtered)
    return [w for w, _ in freq.most_common(top_n)]

def extract_candidate_phrases(passage, correct_answer, max_candidates=20):
    sentences = split_sentences(passage)
    ans_tokens = set(tokenize(correct_answer))
    candidates = set()
    for sent in sentences:
        tokens = remove_stopwords(tokenize(sent))
        for i, t in enumerate(tokens):
            if t not in ans_tokens and len(t) > 3:
                candidates.add(t)
            if i < len(tokens)-1:
                bigram = f"{tokens[i]} {tokens[i+1]}"
                if not any(a in bigram for a in ans_tokens):
                    candidates.add(bigram)
    return list(candidates)[:max_candidates]

def compute_distractor_features(candidates, correct_answer, passage, fe):
    feats = []
    passage_tokens = tokenize(passage)
    passage_freq = Counter(passage_tokens)
    total = max(len(passage_tokens), 1)
    ans_vec     = fe.transform_ohe([clean_text(correct_answer)])
    passage_vec = fe.transform_ohe([clean_text(passage[:500])])
    for cand in candidates:
        cand_vec = fe.transform_ohe([clean_text(cand)])
        sim_ans  = float(cosine_similarity(cand_vec, ans_vec)[0][0])
        char_ov  = len(set(correct_answer.lower()) & set(cand.lower())) / (len(set(correct_answer.lower()))+1e-9)
        freq_s   = sum(passage_freq.get(t,0) for t in tokenize(cand)) / total
        len_s    = min(len(cand.split())/5.0, 1.0)
        sim_pass = float(cosine_similarity(cand_vec, passage_vec)[0][0])
        feats.append([sim_ans, char_ov, freq_s, len_s, sim_pass])
    return np.array(feats, dtype=np.float32)

# ── Distractor Ranker ────────────────────────────────────────────────────────
class DistractorRanker:
    def __init__(self):
        self.model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        self.name = "DistractorRanker_LR"

    def build_training_data(self, df, fe, max_rows=None):
        """Build distractor training data. max_rows=None means use all rows."""
        X_parts, y_parts = [], []
        total = len(df) if max_rows is None else min(max_rows, len(df))
        subset = df.iloc[:total]
        print(f"  Building distractor data from {total} rows...")
        for i, (_, row) in enumerate(subset.iterrows()):
            if i % 1000 == 0 and i > 0:
                print(f"    ... {i}/{total} rows processed")
            passage = str(row.get("article",""))
            ans_letter = row.get("answer","A")
            correct_answer = str(row.get(ans_letter,""))
            wrong_opts = [str(row.get(l,"")) for l in ["A","B","C","D"] if l != ans_letter]
            rand_cands = extract_candidate_words(passage, correct_answer, top_n=5)
            all_cands = wrong_opts + rand_cands
            if not all_cands: continue
            feats = compute_distractor_features(all_cands, correct_answer, passage, fe)
            labels = [1]*len(wrong_opts) + [0]*len(rand_cands)
            labels = labels[:len(feats)]
            X_parts.append(feats); y_parts.extend(labels[:len(feats)])
        if not X_parts: return np.zeros((1,5)), np.array([0])
        return np.vstack(X_parts), np.array(y_parts)

    def fit(self, X, y):
        print(f"  Training {self.name}...")
        self.model.fit(X, y); return self

    def rank_candidates(self, candidates, correct_answer, passage, fe, top_k=3):
        if not candidates: return ["Option A","Option B","Option C"]
        feats = compute_distractor_features(candidates, correct_answer, passage, fe)
        scores = self.model.predict_proba(feats)[:,1]
        sorted_cands = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        ans_tokens = set(tokenize(correct_answer))
        selected, seen = [], set()
        for cand, _ in sorted_cands:
            cand_tokens = set(tokenize(cand))
            overlap = len(cand_tokens & ans_tokens)/(len(cand_tokens)+1e-9)
            if overlap < 0.7 and cand not in seen:
                selected.append(cand); seen.add(cand)
            if len(selected) >= top_k: break
        while len(selected) < top_k:
            selected.append(f"None of the above (option {len(selected)+1})")
        return selected[:top_k]

    def save(self, path=MODEL_B_DIR):
        joblib.dump(self.model, os.path.join(path,"distractor_ranker.pkl"))
        print(f"  Saved {self.name}")

    @staticmethod
    def load(path=MODEL_B_DIR):
        obj = DistractorRanker()
        obj.model = joblib.load(os.path.join(path,"distractor_ranker.pkl"))
        return obj

# ── Hint Generator ───────────────────────────────────────────────────────────
class HintGenerator:
    def __init__(self):
        self.model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        self.name = "HintGenerator_LR"

    def _sent_features(self, sentence, question, correct_answer, position, total):
        kw  = sentence_keyword_overlap(sentence, question)
        pos = position / max(total-1, 1)
        ln  = min(len(sentence.split())/30.0, 1.0)
        ans_present = int(bool(set(tokenize(correct_answer)) & set(tokenize(sentence))))
        q_present   = int(bool(set(remove_stopwords(tokenize(question))) & set(tokenize(sentence))))
        return [kw, pos, ln, ans_present, q_present]

    def build_training_data(self, df, max_rows=None):
        """Build hint training data. max_rows=None means use all rows."""
        X_parts, y_parts = [], []
        total = len(df) if max_rows is None else min(max_rows, len(df))
        subset = df.iloc[:total]
        print(f"  Building hint data from {total} rows...")
        for i, (_, row) in enumerate(subset.iterrows()):
            if i % 1000 == 0 and i > 0:
                print(f"    ... {i}/{total} rows processed")
            passage  = str(row.get("article",""))
            question = str(row.get("question",""))
            correct_answer = str(row.get(row.get("answer","A"),""))
            sents = split_sentences(passage)
            if not sents: continue
            for i_s, sent in enumerate(sents):
                feats = self._sent_features(sent, question, correct_answer, i_s, len(sents))
                kw = sentence_keyword_overlap(sent, question)
                label = 1 if (bool(set(tokenize(correct_answer)) & set(tokenize(sent))) or kw > 0.3) else 0
                X_parts.append(feats); y_parts.append(label)
        if not X_parts: return np.zeros((1,5)), np.array([0])
        return np.array(X_parts, dtype=np.float32), np.array(y_parts)

    def fit(self, X, y):
        print(f"  Training {self.name}...")
        self.model.fit(X, y); return self

    def generate_hints(self, passage, question, correct_answer):
        sents = split_sentences(passage)
        if not sents:
            return {"hint_1":"💡 General: Read the passage carefully.",
                    "hint_2":"🔍 Medium: Focus on the key theme.",
                    "hint_3":"🎯 Specific: Look for the sentence that directly answers the question."}
        scored = []
        for i, sent in enumerate(sents):
            feats = self._sent_features(sent, question, correct_answer, i, len(sents))
            score = self.model.predict_proba([feats])[0][1]
            scored.append((sent, score, i))
        sorted_s = sorted(scored, key=lambda x: x[1], reverse=True)
        n = len(sorted_s)
        early = [s for s in sorted_s if s[2] < len(sents)//2]
        general_src = early[min(1,len(early)-1)] if early else sorted_s[min(1,n-1)]
        medium_src  = sorted_s[0]
        ans_tokens  = set(tokenize(correct_answer))
        specific_src = next((s for s in sorted_s if set(tokenize(s[0])) & ans_tokens), sorted_s[min(1,n-1)])
        return {"hint_1":f"💡 General: {general_src[0]}",
                "hint_2":f"🔍 Medium: {medium_src[0]}",
                "hint_3":f"🎯 Specific: {specific_src[0]}"}

    def save(self, path=MODEL_B_DIR):
        joblib.dump(self.model, os.path.join(path,"hint_generator.pkl"))
        print(f"  Saved {self.name}")

    @staticmethod
    def load(path=MODEL_B_DIR):
        obj = HintGenerator()
        obj.model = joblib.load(os.path.join(path,"hint_generator.pkl"))
        return obj

def evaluate_model_b(ranker, df, fe, max_eval=None):
    """Evaluate distractor ranker. max_eval=None means evaluate on all rows."""
    X, y = ranker.build_training_data(df, fe, max_rows=max_eval)
    if len(y) < 2: return {"accuracy":0.0,"precision":0.0,"recall":0.0,"f1":0.0}
    y_pred = ranker.model.predict(X)
    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y, y_pred)

    print(f"\n  ┌─── DistractorRanker ───")
    print(f"  │  Binary Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  │  Macro Precision   : {prec:.4f}")
    print(f"  │  Macro Recall      : {rec:.4f}")
    print(f"  │  Macro F1-Score    : {f1:.4f}")
    print(f"  └──────────────────────────────")

    return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"confusion_matrix":cm.tolist()}

def train_model_b(data_dir=None, processed_dir=None):
    if data_dir is None: data_dir = DEFAULT_RAW
    if processed_dir is None: processed_dir = DEFAULT_PROCESSED
    print("="*55+"\nMODEL B TRAINING\n"+"="*55)
    tp = os.path.join(processed_dir,"train_expanded.csv")
    if os.path.exists(tp):
        train_exp = pd.read_csv(tp)
        fe = FeatureEngineering.load(MODEL_B_DIR)
    else:
        train_exp, _, _, fe = run_preprocessing_pipeline(data_dir, processed_dir)
    train_raw, val_raw, _ = load_race_dataset(data_dir)
    results = {}

    print("\n[1] Training Distractor Ranker...")
    ranker = DistractorRanker()
    X_d, y_d = ranker.build_training_data(train_raw, fe)   # full dataset
    print(f"  Distractor training data: {X_d.shape}")
    ranker.fit(X_d, y_d); ranker.save()
    results["DistractorRanker"] = evaluate_model_b(ranker, val_raw, fe)

    print("\n[2] Training Hint Generator...")
    hint_gen = HintGenerator()
    X_h_train, y_h_train = hint_gen.build_training_data(train_raw)     # full dataset
    print(f"  Hint training data: {X_h_train.shape}")
    hint_gen.fit(X_h_train, y_h_train); hint_gen.save()
    
    print("\n  Evaluating Hint Generator on Validation Data...")
    X_h_val, y_h_val = hint_gen.build_training_data(val_raw)
    
    if len(y_h_val) > 1:
        yp   = hint_gen.model.predict(X_h_val)
        acc_h  = accuracy_score(y_h_val, yp)
        prec_h = precision_score(y_h_val, yp, average="macro", zero_division=0)
        rec_h  = recall_score(y_h_val, yp, average="macro", zero_division=0)
        f1_h   = f1_score(y_h_val, yp, average="macro", zero_division=0)

        print(f"\n  ┌─── HintGenerator ───")
        print(f"  │  Binary Accuracy   : {acc_h:.4f}  ({acc_h*100:.2f}%)")
        print(f"  │  Macro Precision   : {prec_h:.4f}")
        print(f"  │  Macro Recall      : {rec_h:.4f}")
        print(f"  │  Macro F1-Score    : {f1_h:.4f}")
        print(f"  └──────────────────────────────")

        results["HintGenerator"] = {"accuracy": acc_h, "precision": prec_h,
                                    "recall": rec_h, "f1": f1_h}

    joblib.dump(results, os.path.join(MODEL_B_DIR,"training_results.pkl"))
    print("\n✅ Model B training complete!")
    return results, ranker, hint_gen, fe

if __name__ == "__main__":
    train_model_b(DEFAULT_RAW, DEFAULT_PROCESSED)
