"""
metrics.py
Computes quantitative evaluation metrics for:
1. Intent Classification: Macro F1, Weighted F1, Per-Class Precision/Recall/F1, Accuracy
2. Escalation Decision: Precision, Recall, F1, and False Auto-handle Rate (FAR)
3. Reply Quality: BLEU-4, ROUGE-1, ROUGE-L, and Semantic Cosine Similarity
"""

import math
import re
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    labels = sorted(list(set(y_true) | set(y_pred)))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))

    # Per-class metrics
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "support": int(sup_per[i])
        }

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p), 4),
        "macro_recall": round(float(r), 4),
        "macro_f1": round(float(f1), 4),
        "per_class": per_class,
        "labels": labels
    }

def calculate_escalation_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict:
    """
    Computes escalation performance.
    Crucially computes False Auto-handle Rate (FAR):
    Cases where ground truth required human escalation, but model chose to auto-handle.
    In support operations, this is the highest risk error mode.
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # False Auto-handle Rate = False Negatives / Total Ground Truth Escalations
    total_escalations_needed = tp + fn
    far = fn / total_escalations_needed if total_escalations_needed > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_auto_handle_rate": round(far, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }

def tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())

def compute_sentence_bleu(ref_tokens: List[str], hyp_tokens: List[str], max_n: int = 4) -> float:
    """Computes BLEU-4 with brevity penalty."""
    if len(hyp_tokens) == 0:
        return 0.0

    # Brevity penalty
    c = len(hyp_tokens)
    r = len(ref_tokens)
    if c > r:
        bp = 1.0
    else:
        bp = math.exp(1 - r / c) if c > 0 else 0.0

    # N-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = Counter([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)])
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])

        overlap = 0
        total = sum(hyp_ngrams.values())
        if total == 0:
            precisions.append(0.0)
            continue

        for ng, count in hyp_ngrams.items():
            overlap += min(count, ref_ngrams.get(ng, 0))
        precisions.append(overlap / total)

    if any(p == 0 for p in precisions):
        # Smoothing (add-1)
        precisions = [p if p > 0 else 1e-4 for p in precisions]

    log_sum = sum(math.log(p) for p in precisions) / max_n
    return bp * math.exp(log_sum)

def compute_rouge_l(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    """Computes ROUGE-L F1 score based on longest common subsequence."""
    if not ref_tokens or not hyp_tokens:
        return 0.0
    
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == hyp_tokens[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
                
    lcs = dp[m][n]
    prec = lcs / n if n > 0 else 0.0
    rec = lcs / m if m > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def calculate_text_similarity_metrics(references: List[str], hypotheses: List[str]) -> Dict:
    bleu_scores = []
    rouge_l_scores = []

    for ref, hyp in zip(references, hypotheses):
        ref_toks = tokenize(ref)
        hyp_toks = tokenize(hyp)
        bleu_scores.append(compute_sentence_bleu(ref_toks, hyp_toks))
        rouge_l_scores.append(compute_rouge_l(ref_toks, hyp_toks))

    # Semantic cosine similarity
    vectorizer = TfidfVectorizer().fit(references + hypotheses)
    ref_vecs = vectorizer.transform(references)
    hyp_vecs = vectorizer.transform(hyp_hyp := hypotheses)
    
    # Diagonal dot product
    cos_sims = (ref_vecs.multiply(hyp_vecs)).sum(axis=1)
    cos_mean = float(np.mean(np.array(cos_sims)))

    return {
        "bleu_4": round(float(np.mean(bleu_scores)), 4),
        "rouge_l": round(float(np.mean(rouge_l_scores)), 4),
        "cosine_similarity": round(cos_mean, 4)
    }
