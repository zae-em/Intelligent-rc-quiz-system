"""
inference.py - Unified Inference Pipeline
Loads all trained models and exposes a single API for question generation,
distractor generation, hint generation, and answer verification.
"""
import os, sys, time, random
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import (FeatureEngineering, split_sentences,
                            sentence_keyword_overlap, tokenize, remove_stopwords, clean_text)
from model_a_train import (LogisticRegressionModel, SVMModel, KMeansClusteringModel,
                            EnsembleModel, build_feature_matrix,
                            generate_questions_from_passage, select_best_question, MODEL_A_DIR)
from model_b_train import (DistractorRanker, HintGenerator,
                            extract_candidate_words, extract_candidate_phrases, MODEL_B_DIR)

DATA_PROCESSED = "data/processed"
DATA_RAW       = "data/raw"

class InferencePipeline:
    def __init__(self):
        self.fe=None; self.lr=None; self.svm=None
        self.ensemble=None; self.kmeans=None
        self.ranker=None; self.hint_gen=None
        self.loaded=False; self.session_log=[]

    def load_models(self, auto_train=True):
        print("[InferencePipeline] Loading models...")
        try:
            self.fe       = FeatureEngineering.load(MODEL_A_DIR)
            self.lr       = LogisticRegressionModel.load(MODEL_A_DIR)
            self.svm      = SVMModel.load(MODEL_A_DIR)
            self.ensemble = EnsembleModel.load(self.lr, self.svm, MODEL_A_DIR)
            self.kmeans   = KMeansClusteringModel.load(MODEL_A_DIR)
            self.ranker   = DistractorRanker.load(MODEL_B_DIR)
            self.hint_gen = HintGenerator.load(MODEL_B_DIR)
            self.loaded = True
            print("[InferencePipeline] ✅ All models loaded.")
            return True
        except Exception as e:
            print(f"[InferencePipeline] ⚠️  Could not load: {e}")
            if auto_train:
                return self._auto_train_and_load()
            return False

    def _auto_train_and_load(self):
        try:
            from preprocessing import run_preprocessing_pipeline
            from model_a_train  import train_model_a
            from model_b_train  import train_model_b
            run_preprocessing_pipeline(DATA_RAW, DATA_PROCESSED)
            train_model_a(DATA_RAW, DATA_PROCESSED)
            train_model_b(DATA_RAW, DATA_PROCESSED)
            return self.load_models(auto_train=False)
        except Exception as e:
            print(f"[InferencePipeline] ❌ Auto-training failed: {e}")
            return False

    def generate_question(self, passage, answer=""):
        candidates = generate_questions_from_passage(passage, answer)
        if not candidates:
            return {"question":"What is the main idea of this passage?","source_sentence":passage[:100],"confidence":0.5}
        best_q, best_sent, best_score = select_best_question(candidates, self.fe)
        return {"question":best_q,"source_sentence":best_sent,"confidence":float(best_score)}

    def extract_answer_from_passage(self, passage):
        sentences = split_sentences(passage)
        if not sentences: return "The main topic of the passage."
        for sent in sentences[:3]:
            tokens = remove_stopwords(tokenize(sent))
            if 3 <= len(tokens) <= 10:
                return " ".join(tokens[:5]).capitalize()
        tokens = remove_stopwords(tokenize(sentences[0]))
        return " ".join(tokens[:5]).capitalize() if tokens else sentences[0][:50]

    def generate_distractors(self, passage, question, correct_answer, n=3):
        if not self.loaded:
            return self._fallback_distractors(passage, correct_answer, n)
        word_c   = extract_candidate_words(passage, correct_answer, top_n=30)
        phrase_c = extract_candidate_phrases(passage, correct_answer, max_candidates=20)
        all_c    = list(set(word_c + phrase_c))
        if not all_c:
            return self._fallback_distractors(passage, correct_answer, n)
        return self.ranker.rank_candidates(all_c, correct_answer, passage, self.fe, top_k=n)

    def generate_hints(self, passage, question, correct_answer):
        if not self.loaded:
            return self._fallback_hints(passage, question)
        return self.hint_gen.generate_hints(passage, question, correct_answer)

    def verify_answer(self, passage, question, selected_option, correct_answer):
        is_correct = clean_text(selected_option) == clean_text(correct_answer)
        confidence = 1.0 if is_correct else 0.0
        if self.loaded:
            try:
                row = pd.DataFrame([{"article":passage,"question":question,
                                     "option":selected_option,"label":1,"answer":"A"}])
                X = build_feature_matrix(row, self.fe)
                confidence = float(self.ensemble.predict_proba(X)[0][1])
            except Exception:
                pass
        return {"is_correct":is_correct,"confidence":round(confidence,4),
                "correct_answer":correct_answer,"model_used":"Ensemble(LR+SVM)"}

    def run(self, passage, existing_question=None, existing_answer=None, existing_options=None):
        t0 = time.time()
        if not passage.strip(): raise ValueError("Passage cannot be empty.")
        question = existing_question or self.generate_question(passage, existing_answer or "")["question"]
        correct_answer = existing_answer or self.extract_answer_from_passage(passage)
        if existing_options:
            opts_list = list(existing_options.values()); random.shuffle(opts_list)
        else:
            distractors = self.generate_distractors(passage, question, correct_answer, n=3)
            opts_list = [correct_answer] + distractors; random.shuffle(opts_list)
        letters = ["A","B","C","D"]
        options = {letters[i]: opts_list[i] for i in range(min(4,len(opts_list)))}
        correct_letter = next((l for l,v in options.items() if clean_text(v)==clean_text(correct_answer)), "A")
        hints = self.generate_hints(passage, question, correct_answer)
        elapsed = (time.time()-t0)*1000
        result = {"question":question,"correct_answer":correct_answer,
                  "correct_letter":correct_letter,"options":options,
                  "hints":hints,"inference_time_ms":round(elapsed,2)}
        self.session_log.append({"question":question[:80],"correct_letter":correct_letter,
                                 "inference_time_ms":result["inference_time_ms"]})
        return result

    def get_session_log(self):
        if not self.session_log:
            return pd.DataFrame(columns=["question","correct_letter","inference_time_ms"])
        return pd.DataFrame(self.session_log)

    def get_model_a_metrics(self):
        p = os.path.join(MODEL_A_DIR,"training_results.pkl")
        return joblib.load(p) if os.path.exists(p) else {}

    def get_model_b_metrics(self):
        p = os.path.join(MODEL_B_DIR,"training_results.pkl")
        return joblib.load(p) if os.path.exists(p) else {}

    def _fallback_distractors(self, passage, correct_answer, n=3):
        words = extract_candidate_words(passage, correct_answer, top_n=n+5)
        pad   = ["This is not mentioned","None of the above","Cannot be determined"]
        return (words + pad)[:n]

    def _fallback_hints(self, passage, question):
        sents = split_sentences(passage)
        if not sents:
            return {"hint_1":"💡 General: Read the passage carefully.",
                    "hint_2":"🔍 Medium: Focus on key sentences.",
                    "hint_3":"🎯 Specific: Find the sentence directly related to the question."}
        scored = sorted([(s, sentence_keyword_overlap(s,question)) for s in sents],
                        key=lambda x: x[1], reverse=True)
        h1 = sents[len(sents)//2] if len(sents)>2 else sents[0]
        return {"hint_1":f"💡 General: {h1}",
                "hint_2":f"🔍 Medium: {scored[min(1,len(scored)-1)][0]}",
                "hint_3":f"🎯 Specific: {scored[0][0]}"}

_singleton = None
def get_pipeline(auto_train=True):
    global _singleton
    if _singleton is None:
        _singleton = InferencePipeline()
        _singleton.load_models(auto_train=auto_train)
    return _singleton

if __name__ == "__main__":
    passage = ("The Amazon rainforest is the world's largest tropical rainforest covering most "
               "of the Amazon basin of South America. It represents over half of the planet's "
               "remaining rainforests and is home to three million species. Deforestation "
               "driven by cattle ranching remains a critical threat to this ecosystem.")
    pipe = InferencePipeline()
    pipe.load_models(auto_train=True)
    result = pipe.run(passage)
    print("\n== RESULT ==")
    print(f"Q: {result['question']}")
    for l,v in result['options'].items():
        mark = " ✓" if l==result['correct_letter'] else ""
        print(f"  {l}. {v}{mark}")
    for k,v in result['hints'].items(): print(f"  {k}: {v}")
    print(f"Time: {result['inference_time_ms']} ms")
