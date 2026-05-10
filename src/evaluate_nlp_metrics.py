import os
import sys
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_race_dataset, DEFAULT_RAW
from model_a_train import generate_questions_from_passage, select_best_question

DATA_RAW = DEFAULT_RAW

def approximate_meteor(ref_tokens, gen_tokens):
    # Simplified METEOR without wordnet (exact match precision/recall + penalty)
    if not ref_tokens or not gen_tokens: return 0.0
    matches = sum(1 for t in gen_tokens if t in ref_tokens)
    if matches == 0: return 0.0
    precision = matches / len(gen_tokens)
    recall = matches / len(ref_tokens)
    fmean = (10 * precision * recall) / (recall + 9 * precision)
    # Basic penalty for chunks
    penalty = 0.5 * (1**3) # Simplified
    return fmean * (1 - penalty)

def evaluate_generation_metrics():
    print("="*55)
    print("NLP GENERATION METRICS EVALUATION")
    print("="*55)
    
    # Load dataset
    _, val_raw, test_raw = load_race_dataset(DATA_RAW)
    df = test_raw if not test_raw.empty else val_raw
    if df.empty:
        print("No test/val data available for evaluation.")
        return

    subset_size = min(50, len(df))
    df_subset = df.iloc[:subset_size]
    print(f"Evaluating text generation (Question Generation) on {subset_size} samples...\n")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4

    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    meteor_scores = []

    for i, row in df_subset.iterrows():
        passage = str(row.get("article", ""))
        ref_question = str(row.get("question", ""))
        answer_letter = str(row.get("answer", "A"))
        correct_answer = str(row.get(answer_letter, ""))

        candidates = generate_questions_from_passage(passage, correct_answer)
        gen_q, _, _ = select_best_question(candidates)

        try:
            ref_tokens = nltk.word_tokenize(ref_question.lower())
            gen_tokens = nltk.word_tokenize(gen_q.lower())
        except LookupError:
            # Fallback tokenizer
            ref_tokens = ref_question.lower().split()
            gen_tokens = gen_q.lower().split()

        # 1. BLEU Score
        bleu = sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu_scores.append(bleu)

        # 2. ROUGE Score
        scores = scorer.score(ref_question, gen_q)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

        # 3. METEOR Score fallback
        meteor_scores.append(approximate_meteor(ref_tokens, gen_tokens))

    print("--- GENERATION METRICS ---")
    print(f"Average BLEU-1 Score: {sum(bleu_scores)/len(bleu_scores):.4f}")
    print(f"Average ROUGE-1     : {sum(rouge1_scores)/len(rouge1_scores):.4f}")
    print(f"Average ROUGE-2     : {sum(rouge2_scores)/len(rouge2_scores):.4f}")
    print(f"Average ROUGE-L     : {sum(rougeL_scores)/len(rougeL_scores):.4f}")
    print(f"Average METEOR Score: {sum(meteor_scores)/len(meteor_scores):.4f} (Approximated)")
    print("\n Generation metrics calculated successfully.")

if __name__ == "__main__":
    evaluate_generation_metrics()
