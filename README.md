# 📚 Intelligent Reading Comprehension & Quiz Generation System

> BS (CS) Spring 2026 — Artificial Intelligence Project  
> National University of Computer and Emerging Sciences (FAST), Islamabad

---

## 🧠 Project Overview

An end-to-end AI-powered Reading Comprehension system built on the **RACE dataset** using **Traditional ML** (no neural networks required). The system:

- Takes a reading passage as input
- Generates a multiple-choice question (template-based + ML ranking)
- Produces 1 correct answer + 3 plausible distractors
- Verifies user answers using a trained ensemble classifier
- Generates 3 graduated hints (General → Medium → Specific)
- Displays analytics via a polished Streamlit UI

---

## 🗂 Project Structure

```
race_rc_project/
├── data/
│   ├── raw/                  # Place train.csv, val.csv, test.csv here
│   └── processed/            # Auto-generated feature matrices & plots
├── models/
│   ├── model_a/traditional/  # LR, SVM, KMeans, Ensemble weights
│   └── model_b/traditional/  # DistractorRanker, HintGenerator
├── src/
│   ├── preprocessing.py      # Text cleaning, OHE, TF-IDF, feature engineering
│   ├── model_a_train.py      # LR, SVM, KMeans, Template QGen, Ensemble
│   ├── model_b_train.py      # Distractor Ranker, Hint Generator
│   ├── inference.py          # Unified inference API
│   └── evaluate.py           # Metrics: Acc, P, R, F1, CM, Silhouette, R²
├── ui/
│   └── app.py                # Streamlit UI (4 screens)
├── notebooks/
│   └── EDA.ipynb             # Exploratory Data Analysis
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone / Download

```bash
git clone <your-repo-url>
cd race_rc_project
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Setup

Download the **RACE dataset** from Kaggle:  
🔗 https://www.kaggle.com/datasets/swaptr/race-dataset

Place the CSV files inside `data/raw/`:
```
data/raw/train.csv
data/raw/val.csv
data/raw/test.csv
```

Expected columns: `id, article, question, A, B, C, D, answer`

> **Note:** If CSV files are not found, the system automatically generates synthetic demo data so you can test the full pipeline immediately.

---

## 🚀 How to Run

### Option A — Run Full Pipeline (Recommended)

```bash
cd race_rc_project

# Step 1: Preprocessing
python src/preprocessing.py

# Step 2: Train Model A
python src/model_a_train.py

# Step 3: Train Model B
python src/model_b_train.py

# Step 4: Evaluate
python src/evaluate.py

# Step 5: Launch UI
streamlit run ui/app.py
```

### Option B — Auto-train on First Launch

The Streamlit app will auto-train all models on first launch if they are not found:

```bash
cd race_rc_project
streamlit run ui/app.py
```

---

## 🖥 UI Screens

| Screen | Description |
|--------|-------------|
| 📖 **Article Input** | Paste passage, upload .txt, or load RACE sample |
| 🧠 **Quiz View** | MCQ with 4 options, check answer, color-coded feedback |
| 💡 **Hint Panel** | 3 graduated hints (General → Medium → Specific), reveal answer |
| 📊 **Analytics** | Model A & B metrics, confusion matrices, session log, CSV export |

---

## 🤖 Models

### Model A — Question & Answer Verifier
| Model | Features | Task |
|-------|----------|------|
| Logistic Regression | OHE(article+question+option) + lexical + cosine | Answer verification |
| SVM (LinearSVC + Calibration) | Same as LR | Answer verification |
| K-Means Clustering | OHE + SVD | Unsupervised pattern discovery |
| Ensemble (Soft Voting) | LR + SVM probability average | Answer verification |
| Template QGen | Rule-based Wh-word templates + ML ranking | Question generation |

### Model B — Distractor & Hint Generator
| Model | Task |
|-------|------|
| Distractor Ranker (LR) | Score & rank candidate distractors from passage |
| Hint Generator (LR) | Score sentences for extractive hint generation |

---

## 📊 Evaluation Metrics

- **Model A:** Accuracy, Precision, Recall, F1, Confusion Matrix, Silhouette Score
- **Model B:** Accuracy, Precision, Recall, F1, Confusion Matrix, R² Score

---

## 📦 Dependencies

```
streamlit>=1.32.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
scipy>=1.11.0
```

---

## 📸 Screenshots

> *(Add screenshots of your running app here)*
>
> - `screenshots/screen1_input.png`
> - `screenshots/screen2_quiz.png`
> - `screenshots/screen3_hints.png`
> - `screenshots/screen4_analytics.png`

---

## ⚠️ Ethical Considerations

- AI-generated questions may contain errors — always review before use in real assessments
- The RACE dataset originates from Chinese school exams — consider cultural/linguistic bias in model outputs
- The UI clearly indicates AI-generated content
- Not intended for deployment in real exam settings without human review

---

## 📚 References

1. Lai et al. (2017). RACE: Large-scale ReAding Comprehension Dataset From Examinations. *EMNLP 2017*.
2. Du et al. (2017). Learning to Ask. *ACL 2017*.
3. Guo et al. (2016). Generating Distractors for Reading Comprehension Questions. *AAAI 2016*.
4. Devlin et al. (2019). BERT. *NAACL 2019*.
5. Papineni et al. (2002). BLEU. *ACL 2002*.

---

*Built with ❤️ using Python, scikit-learn, and Streamlit.*
