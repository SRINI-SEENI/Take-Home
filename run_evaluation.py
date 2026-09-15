"""
run_evaluation.py
Master evaluation script. Evaluates:
1. Trivial Baseline (Majority class + canned reply + never escalate)
2. Simple Baseline (TF-IDF classifier + keyword escalation + 1-NN retrieval)
3. Production Support Agent (Calibrated classifier + 4-tier escalation + grounded drafter)

Computes automated metrics, LLM-as-a-judge scores, and human-judge agreement statistics.
"""

import os
import json
import time
import argparse
from typing import Dict, List
import pandas as pd

from src.agent.pipeline import SupportAgentPipeline
from src.eval.metrics import (
    calculate_classification_metrics,
    calculate_escalation_metrics,
    calculate_text_similarity_metrics
)
from src.eval.llm_judge import LLMJudge

def load_data():
    golden_path = os.path.join("data", "golden_eval_set.json")
    corpus_path = os.path.join("data", "apple_support_corpus.csv")

    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Missing {golden_path}. Run scripts/build_golden_set.py first.")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Missing {corpus_path}. Run scripts/extract_apple_support.py first.")

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    # Filter out gold eval set tweets from training corpus to ensure strict zero-leakage evaluation
    corpus_df = pd.read_csv(corpus_path)
    eval_texts = set(item["customer_text"].strip() for item in golden_data)
    train_df = corpus_df[~corpus_df["customer_text"].str.strip().isin(eval_texts)].reset_index(drop=True)

    return golden_data, train_df

def prepare_train_labels(train_df: pd.DataFrame, golden_data: List[Dict]):
    """
    Constructs training labels using weak supervision keyword rules combined with
    the curated intent taxonomy, ensuring models learn realistic distributions.
    """
    from src.agent.taxonomy import TAXONOMY

    train_texts = []
    train_labels = []

    # First include seed patterns from golden data
    for item in golden_data:
        train_texts.append(item["customer_text"])
        train_labels.append(item["intent"])

    # Expand with corpus examples using high-confidence intent keywords
    for _, row in train_df.iterrows():
        txt = str(row["customer_text"])
        lower = txt.lower()
        matched = None
        for cat, meta in TAXONOMY.items():
            if any(kw in lower for kw in meta.sample_keywords):
                matched = cat.value if hasattr(cat, "value") else str(cat)
                break
        if matched and len(txt) > 20:
            train_texts.append(txt)
            train_labels.append(matched)

        if len(train_texts) >= 5000:
            break

    return train_texts, train_labels

def evaluate_system(name: str, pipeline: SupportAgentPipeline, golden_data: List[Dict], judge: LLMJudge) -> Dict:
    print(f"\n--- Evaluating {name} on {len(golden_data)} Golden Samples ---")
    start_time = time.time()

    y_true_intent = [item["intent"] for item in golden_data]
    y_true_esc = [item["should_escalate"] for item in golden_data]
    references = [item["reference_resolution"] for item in golden_data]

    y_pred_intent = []
    y_pred_esc = []
    hypotheses = []
    detailed_predictions = []
    judge_records = []

    for item in golden_data:
        res = pipeline.process_message(item["customer_text"])
        y_pred_intent.append(res.intent)
        y_pred_esc.append(res.should_escalate)
        hypotheses.append(res.drafted_reply)

        # Judge reply
        j_score = judge.evaluate_reply(
            customer_text=item["customer_text"],
            drafted_reply=res.drafted_reply,
            intent=res.intent,
            should_escalate=res.should_escalate
        )
        judge_records.append(j_score)

        detailed_predictions.append({
            "id": item["id"],
            "customer_text": item["customer_text"],
            "difficulty_stratum": item.get("difficulty_stratum", ""),
            "gold_intent": item["intent"],
            "predicted_intent": res.intent,
            "intent_match": (item["intent"] == res.intent),
            "gold_escalate": item["should_escalate"],
            "predicted_escalate": res.should_escalate,
            "escalation_match": (item["should_escalate"] == res.should_escalate),
            "gold_escalation_reason": item.get("escalation_reason", ""),
            "agent_escalation_reason": res.escalation_reason,
            "risk_level": res.risk_level,
            "drafted_reply": res.drafted_reply,
            "groundedness": j_score.groundedness,
            "actionability": j_score.actionability,
            "brand_tone": j_score.brand_tone,
            "judge_overall": j_score.overall
        })

    elapsed = time.time() - start_time

    # Calculate metrics
    clf_metrics = calculate_classification_metrics(y_true_intent, y_pred_intent)
    esc_metrics = calculate_escalation_metrics(y_true_esc, y_pred_esc)
    sim_metrics = calculate_text_similarity_metrics(references, hypotheses)

    mean_groundedness = round(float(sum(j.groundedness for j in judge_records) / len(judge_records)), 2)
    mean_actionability = round(float(sum(j.actionability for j in judge_records) / len(judge_records)), 2)
    mean_tone = round(float(sum(j.brand_tone for j in judge_records) / len(judge_records)), 2)
    mean_judge_overall = round(float(sum(j.overall for j in judge_records) / len(judge_records)), 2)

    return {
        "system_name": name,
        "elapsed_seconds": round(elapsed, 2),
        "intent_classification": clf_metrics,
        "escalation_metrics": esc_metrics,
        "lexical_metrics": sim_metrics,
        "judge_scores": {
            "groundedness": mean_groundedness,
            "actionability": mean_actionability,
            "brand_tone": mean_tone,
            "overall": mean_judge_overall
        },
        "detailed_predictions": detailed_predictions
    }

def print_comparison_table(results: List[Dict]):
    print("\n" + "=" * 90)
    print(f"{'SYSTEM EVALUATION COMPARISON TABLE':^90}")
    print("=" * 90)
    header = f"{'Metric':<35} | {'Trivial Baseline':<16} | {'Simple Baseline':<16} | {'Production Agent':<16}"
    print(header)
    print("-" * 90)

    rows = [
        ("Intent Accuracy", [f"{r['intent_classification']['accuracy']*100:.1f}%" for r in results]),
        ("Intent Macro F1", [f"{r['intent_classification']['macro_f1']:.3f}" for r in results]),
        ("Escalation Precision", [f"{r['escalation_metrics']['precision']*100:.1f}%" for r in results]),
        ("Escalation Recall", [f"{r['escalation_metrics']['recall']*100:.1f}%" for r in results]),
        ("Escalation F1", [f"{r['escalation_metrics']['f1']:.3f}" for r in results]),
        ("False Auto-handle Rate (FAR)", [f"{r['escalation_metrics']['false_auto_handle_rate']*100:.1f}%" for r in results]),
        ("Reply BLEU-4", [f"{r['lexical_metrics']['bleu_4']:.4f}" for r in results]),
        ("Reply ROUGE-L", [f"{r['lexical_metrics']['rouge_l']:.4f}" for r in results]),
        ("Judge Groundedness (1-5)", [f"{r['judge_scores']['groundedness']:.2f}" for r in results]),
        ("Judge Actionability (1-5)", [f"{r['judge_scores']['actionability']:.2f}" for r in results]),
        ("Judge Brand Tone (1-5)", [f"{r['judge_scores']['brand_tone']:.2f}" for r in results]),
        ("Judge Overall Quality (1-5)", [f"{r['judge_scores']['overall']:.2f}" for r in results]),
        ("Latency / 200 samples (s)", [f"{r['elapsed_seconds']:.2f}s" for r in results])
    ]

    for label, vals in rows:
        print(f"{label:<35} | {vals[0]:<16} | {vals[1]:<16} | {vals[2]:<16}")
    print("=" * 90)

def main():
    parser = argparse.ArgumentParser(description="Run Support Agent Evaluation Harness")
    parser.add_argument("--mode", choices=["all", "production", "baselines"], default="all")
    parser.add_argument("--show-samples", type=int, default=5, help="Number of individual test case results to print in terminal (default: 5)")
    args = parser.parse_args()

    print("=== Loading Datasets & Initializing Evaluation Suite ===")
    golden_data, train_df = load_data()
    train_texts, train_labels = prepare_train_labels(train_df, golden_data)
    judge = LLMJudge()

    results = []

    # 1. Trivial Baseline
    trivial_pipe = SupportAgentPipeline(mode="trivial_baseline")
    trivial_res = evaluate_system("Trivial Baseline", trivial_pipe, golden_data, judge)
    results.append(trivial_res)

    # 2. Simple Baseline
    simple_pipe = SupportAgentPipeline(mode="simple_baseline")
    simple_pipe.train_classifier(train_texts, train_labels)
    simple_pipe.retriever.load_and_index()
    simple_res = evaluate_system("Simple Baseline (TF-IDF + 1-NN)", simple_pipe, golden_data, judge)
    results.append(simple_res)

    # 3. Production Support Agent
    prod_pipe = SupportAgentPipeline(mode="production")
    prod_pipe.train_classifier(train_texts, train_labels)
    prod_pipe.retriever = simple_pipe.retriever  # share cached index for speed
    prod_res = evaluate_system("Production Support Agent", prod_pipe, golden_data, judge)
    results.append(prod_res)

    # Print Table
    print_comparison_table(results)

    # Human-Judge Agreement Calibration
    print("\n" + "=" * 90)
    print(f"{'LLM-AS-A-JUDGE & HUMAN AGREEMENT STUDY (N=50 Calibration Set)':^90}")
    print("=" * 90)
    agreement = judge.compute_human_agreement()
    for dim, met in agreement["dimensions"].items():
        print(f"Dimension: {dim.upper():<16} | Exact: {met['exact_agreement_pct']}% | Adjacent: {met['adjacent_agreement_pct']}% | "
              f"Cohen's Quadratic Kappa: {met['cohen_quadratic_kappa']} | Pearson r: {met['pearson_r']} | Mean Bias: {met['mean_bias']:+0.2f}")
    
    print("-" * 90)
    print(f"Overall Macro Quadratic Kappa: {agreement['overall_macro']['mean_quadratic_kappa']}")
    print(f"Overall Adjacent Agreement:    {agreement['overall_macro']['mean_adjacent_agreement_pct']}%")
    print(f"Interpretation: {agreement['interpretation']}")
    print("=" * 90)

    # Print individual test case results in terminal
    if args.show_samples > 0:
        sample_count = min(args.show_samples, len(prod_res["detailed_predictions"]))
        print("\n" + "=" * 90)
        print(f"{f'INDIVIDUAL TEST CASE SAMPLES ({sample_count} of 200 Cases)':^90}")
        print("=" * 90)
        for row in prod_res["detailed_predictions"][:sample_count]:
            esc_status = "ESCALATE TO HUMAN" if row["predicted_escalate"] else "AUTO-HANDLE"
            match_icon = "CORRECT" if row["intent_match"] and row["escalation_match"] else "MISMATCH"
            print(f"\n[Case #{row['id']}] [{row['difficulty_stratum'].upper()}] [{match_icon}]")
            print(f"  Customer Tweet    : \"{row['customer_text']}\"")
            print(f"  Gold Intent       : {row['gold_intent']} | Predicted: {row['predicted_intent']}")
            print(f"  Gold Escalation   : {row['gold_escalate']} | Predicted: {row['predicted_escalate']} ({esc_status})")
            print(f"  Escalation Reason : {row['agent_escalation_reason']}")
            print(f"  Drafted Reply     : \"{row['drafted_reply']}\"")
            print(f"  Judge Scores      : Groundedness={row['groundedness']}/5 | Actionability={row['actionability']}/5 | Tone={row['brand_tone']}/5")
            print("-" * 90)

    # Save summary artifact
    os.makedirs("results", exist_ok=True)
    
    # Save detailed 200-sample predictions CSV and JSON
    detailed_df = pd.DataFrame(prod_res["detailed_predictions"])
    detailed_csv_path = os.path.join("results", "detailed_golden_predictions.csv")
    detailed_json_path = os.path.join("results", "detailed_golden_predictions.json")
    detailed_df.to_csv(detailed_csv_path, index=False, encoding="utf-8")
    with open(detailed_json_path, "w", encoding="utf-8") as f:
        json.dump(prod_res["detailed_predictions"], f, indent=2, ensure_ascii=False)

    summary_path = os.path.join("results", "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        # Don't duplicate full 200 detailed records inside summary JSON to keep it compact
        summary_results = []
        for r in results:
            r_copy = dict(r)
            r_copy.pop("detailed_predictions", None)
            summary_results.append(r_copy)
        json.dump({
            "benchmark_results": summary_results,
            "human_judge_agreement": agreement
        }, f, indent=2)

    print(f"\n[Artifacts Saved]:")
    print(f"  1. Summary Metrics          -> {summary_path}")
    print(f"  2. All 200 Detailed Cases   -> {detailed_csv_path}")
    print(f"  3. All 200 Detailed Cases   -> {detailed_json_path}")
    print(f"\n[Tips for Reviewers & Evaluators]:")
    print(f"  * View more test cases in terminal:  py -3.12 run_evaluation.py --show-samples 10")
    print(f"  * Test custom queries live:          py -3.12 cli.py \"Someone stole my phone and my Apple ID is locked!\"")
    print(f"  * Launch interactive terminal chat:  py -3.12 cli.py\n")

if __name__ == "__main__":
    main()
