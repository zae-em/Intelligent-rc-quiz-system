"""
preprocessing.py - Dataset loading, text cleaning, tokenization, feature engineering.
Primary feature: One-Hot Encoding (binary CountVectorizer). TF-IDF optional.
"""
import os, re, string, pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Project root = one level above this src/ file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAW       = os.path.join(BASE_DIR, "data", "raw")
DEFAULT_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
DEFAULT_FE_DIR    = os.path.join(BASE_DIR, "models", "model_a", "traditional")

STOPWORDS = set([
    "a","an","the","is","it","in","on","at","to","for","of","and","or","but",
    "so","with","as","by","from","that","this","was","are","be","been","have",
    "has","had","do","does","did","not","no","can","will","would","could",
    "should","may","might","i","he","she","we","they","you","my","his","her",
    "our","their","its","your","who","which","what","when","where","why","how"
])
MAX_OHE = 5000
MAX_TFIDF = 3000
  
def clean_text(text):
    if not isinstance(text, str): text = str(text)
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    return re.sub(r"\s+"," ",text).strip()

def tokenize(text):
    return clean_text(text).split()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]

def split_sentences(text):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 10]

def sentence_keyword_overlap(sentence, query):
    s = set(remove_stopwords(tokenize(sentence)))
    q = set(remove_stopwords(tokenize(query)))
    return len(s & q) / len(q) if q else 0.0

def word_frequency(tokens):
    return Counter(tokens)

def load_race_dataset(data_dir=None, chunksize=None):
    if data_dir is None:
        data_dir = DEFAULT_RAW
    """Load train/val/test CSVs. Use chunksize to read in chunks for very large files."""
    splits = {}
    for split in ["train","val","test"]:
        path = os.path.join(data_dir, f"{split}.csv")
        if os.path.exists(path):
            if chunksize:
                chunks = pd.read_csv(path, chunksize=chunksize)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(path, low_memory=False, nrows=4000)
            splits[split] = df
            print(f"  Loaded {split}: {df.shape}")
        else:
            print(f"  [{split}.csv not found] generating synthetic demo data")
            splits[split] = _generate_synthetic(200 if split=="train" else 50)
    return splits["train"], splits["val"], splits["test"]

def _generate_synthetic(n=200):
    np.random.seed(42)
    articles = [
        "The Amazon rainforest is the world's largest tropical rainforest covering most of the Amazon basin. It represents over half of the planet's remaining rainforests and is home to three million species. Deforestation remains a critical threat driven by cattle ranching and agriculture.",
        "The human brain weighs about 1.4 kilograms and contains approximately 86 billion neurons. The cerebrum controls thinking memory and voluntary movement. The cerebellum coordinates balance. Scientists continue discovering new facts about brain information processing.",
        "Solar energy converts sunlight into electrical energy using photovoltaic cells in solar panels. It is the cleanest most abundant renewable energy source. Solar costs have dropped dramatically over the past decade making it accessible worldwide.",
        "The Great Wall of China stretches over 21196 kilometers across northern China. It was built to protect Chinese states from northern invasions beginning as early as the 7th century BC. It remains one of the greatest architectural feats in history.",
        "Photosynthesis is the process by which plants use sunlight water and carbon dioxide to produce oxygen and sugar. It takes place in chloroplasts inside plant leaves. Chlorophyll absorbs sunlight and is fundamental to all life on Earth."
    ]
    questions = [
        "What is the main threat to the Amazon rainforest?",
        "What does the cerebrum control?",
        "What converts sunlight into electricity in solar panels?",
        "How long is the Great Wall of China?",
        "Where does photosynthesis primarily take place?"
    ]
    correct = ["Deforestation","Thinking memory and voluntary movement","Photovoltaic cells","21196 kilometers","In chloroplasts inside leaves"]
    distractors = [
        ["Flooding","Tourism","Earthquakes"],
        ["Breathing and digestion","Body temperature","Eye movement"],
        ["Wind turbines","Coal generators","Hydroelectric dams"],
        ["5000 kilometers","10000 kilometers","50000 kilometers"],
        ["In the roots","In the stem","In the flowers"]
    ]
    rows = []
    for i in range(n):
        idx = i % len(articles)
        ans_letter = ["A","B","C","D"][i % 4]
        opts = distractors[idx].copy()
        opts.insert(["A","B","C","D"].index(ans_letter), correct[idx])
        rows.append({"id":f"syn_{i}","article":articles[idx],"question":questions[idx],
                     "A":opts[0],"B":opts[1],"C":opts[2],"D":opts[3],"answer":ans_letter})
    return pd.DataFrame(rows)

def expand_options(df):
    """Vectorized expand: each row becomes 4 rows (one per option A-D)."""
    frames = []
    for letter in ["A", "B", "C", "D"]:
        tmp = df[["id", "article", "question", letter, "answer"]].copy()
        tmp = tmp.rename(columns={letter: "option"})
        tmp["option"] = tmp["option"].astype(str)
        tmp["label"] = (tmp["answer"] == letter).astype(int)
        frames.append(tmp)
    result = pd.concat(frames, ignore_index=True)
    return result

class FeatureEngineering:
    def __init__(self):
        self.ohe = CountVectorizer(max_features=MAX_OHE, binary=True, ngram_range=(1,2), min_df=1)
        self.tfidf = TfidfVectorizer(max_features=MAX_TFIDF, ngram_range=(1,2), min_df=1)
        self.fitted = False

    def fit(self, texts):
        cleaned = [clean_text(t) for t in texts]
        self.ohe.fit(cleaned); self.tfidf.fit(cleaned)
        self.fitted = True
        print(f"  OHE vocab: {len(self.ohe.vocabulary_)}  TF-IDF vocab: {len(self.tfidf.vocabulary_)}")

    def transform_ohe(self, texts):
        return self.ohe.transform([clean_text(t) for t in texts])

    def transform_tfidf(self, texts):
        return self.tfidf.transform([clean_text(t) for t in texts])

    def lexical_features(self, df_exp):
        """Vectorized lexical features — avoids slow iterrows on large datasets."""
        def _overlap(set_a, set_b):
            return len(set_a & set_b) / (len(set_b) + 1e-9)

        # fillna("") guards against float NaN values read back from CSV
        articles  = df_exp["article"].fillna("").astype(str).tolist()
        questions = df_exp["question"].fillna("").astype(str).tolist()
        options   = df_exp["option"].fillna("").astype(str).tolist()

        a_tok = [set(remove_stopwords(tokenize(t))) for t in articles]
        q_tok = [set(remove_stopwords(tokenize(t))) for t in questions]
        o_tok = [set(remove_stopwords(tokenize(t))) for t in options]

        len_a  = np.array([len(a) for a in a_tok], dtype=np.float32)
        len_q  = np.array([len(q) for q in q_tok], dtype=np.float32)
        len_o  = np.array([len(o) for o in o_tok], dtype=np.float32)
        ao_ov  = np.array([_overlap(a, o) for a, o in zip(a_tok, o_tok)], dtype=np.float32)
        qo_ov  = np.array([_overlap(q, o) for q, o in zip(q_tok, o_tok)], dtype=np.float32)
        cap    = np.array([int(str(opt)[:1].isupper()) for opt in options], dtype=np.float32)

        return np.column_stack([len_a, len_q, len_o, ao_ov, qo_ov, cap])

    def cosine_features(self, df_exp):
        av = self.transform_ohe(df_exp["article"].tolist())
        ov = self.transform_ohe(df_exp["option"].tolist())
        qv = self.transform_ohe(df_exp["question"].tolist())
        ao = np.array([cosine_similarity(av[i], ov[i])[0][0] for i in range(av.shape[0])]).reshape(-1,1)
        qo = np.array([cosine_similarity(qv[i], ov[i])[0][0] for i in range(qv.shape[0])]).reshape(-1,1)
        return np.hstack([ao, qo])

    def save(self, path="models/model_a/traditional"):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path,"feature_engineering.pkl"),"wb") as f:
            pickle.dump(self, f)
        print(f"  FE saved → {path}")

    @staticmethod
    def load(path="models/model_a/traditional"):
        with open(os.path.join(path,"feature_engineering.pkl"),"rb") as f:
            return pickle.load(f)

def run_preprocessing_pipeline(data_dir=None, out_dir=None, fe_dir=None):
    if data_dir is None: data_dir = DEFAULT_RAW
    if out_dir  is None: out_dir  = DEFAULT_PROCESSED
    if fe_dir   is None: fe_dir   = DEFAULT_FE_DIR
    model_b_dir = os.path.join(BASE_DIR, "models", "model_b", "traditional")
    print("="*55+"\nPREPROCESSING PIPELINE\n"+"="*55)
    train_df, val_df, test_df = load_race_dataset(data_dir)
    print("  Expanding options (vectorized)...")
    train_exp = expand_options(train_df)
    val_exp   = expand_options(val_df)
    test_exp  = expand_options(test_df)
    print(f"  Expanded → train:{len(train_exp)} val:{len(val_exp)} test:{len(test_exp)} rows")

    # Fit vectorizers on a sample for speed if dataset is very large (>200K combined texts)
    combined = (train_exp["article"]+" "+train_exp["question"]+" "+train_exp["option"]).tolist()
    if len(combined) > 200_000:
        print(f"  Large dataset ({len(combined)} texts) — fitting vectorizer on 100K sample...")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(combined), size=100_000, replace=False)
        fit_texts = [combined[i] for i in idx]
    else:
        fit_texts = combined

    fe = FeatureEngineering()
    fe.fit(fit_texts)
    os.makedirs(out_dir, exist_ok=True)
    print("  Saving expanded CSVs...")
    train_exp.to_csv(os.path.join(out_dir,"train_expanded.csv"), index=False)
    val_exp.to_csv(os.path.join(out_dir,"val_expanded.csv"), index=False)
    test_exp.to_csv(os.path.join(out_dir,"test_expanded.csv"), index=False)
    fe.save(fe_dir)
    fe.save(model_b_dir)
    print("✅ Preprocessing complete!")
    return train_exp, val_exp, test_exp, fe

if __name__ == "__main__":
    run_preprocessing_pipeline(
        data_dir=DEFAULT_RAW,
        out_dir=DEFAULT_PROCESSED,
        fe_dir=DEFAULT_FE_DIR
    )
