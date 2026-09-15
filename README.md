# Production AI Customer Support Agent (@AppleSupport)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-take--home.streamlit.app-brightgreen?logo=streamlit)](https://take-home.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reproducibility](https://img.shields.io/badge/reproducibility-%3C15s-brightgreen.svg)]()
[![Evaluation](https://img.shields.io/badge/Golden%20Eval%20Set-200%20Samples-orange.svg)]()

A production-grade AI customer support system for **`@AppleSupport`**, developed for the **Hiver SDE Intern Take-Home Assignment**.

The system:
1. **Classifies incoming customer tweets** into an 8-class domain-specific taxonomy.
2. **Drafts resolution replies** grounded in 14,706 historically verified Apple Support resolutions.
3. **Makes auditable escalation decisions** (Auto-handle vs. Escalate to Human) with machine-readable triggers and human-readable rationales.
4. **Evaluates performance** against two baselines (Trivial & Simple) using automated metrics, an LLM-as-a-Judge rubric, and an empirical **Human-Judge Agreement study** (Cohen's Quadratic Weighted Kappa: **0.9068**).

Full engineering analysis, failure modes, and the mandatory *"What is misleading about my headline number?"* section are documented in [REPORT.md](REPORT.md).

---

## 🚀 Quickstart: Reproduce Headline Results in < 15 Seconds

No external API keys or Docker containers are required to reproduce the headline results.

### 1. Prerequisites
- Python 3.10+ installed
- Git

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Evaluation Suite
```bash
# Runs evaluation across all 3 systems + LLM judge calibration
python run_evaluation.py
```
*(On Windows using the Python launcher, use `py -3.12 run_evaluation.py`)*

### 4. Expected Terminal Output
You will see the complete benchmark comparison table and human agreement calibration in under 3 seconds:
```text
==========================================================================================
                            SYSTEM EVALUATION COMPARISON TABLE                            
==========================================================================================
Metric                              | Trivial Baseline | Simple Baseline  | Production Agent
------------------------------------------------------------------------------------------
Intent Accuracy                     | 2.5%             | 99.0%            | 98.5%           
Intent Macro F1                     | 0.005            | 0.899            | 0.865           
Escalation Precision                | 0.0%             | 100.0%           | 81.2%           
Escalation Recall                   | 0.0%             | 25.0%            | 81.2%           
Escalation F1                       | 0.000            | 0.400            | 0.812           
False Auto-handle Rate (FAR)        | 100.0%           | 75.0%            | 18.8%           
Reply BLEU-4                        | 0.0019           | 0.0010           | 0.1460          
Reply ROUGE-L                       | 0.0715           | 0.1285           | 0.3437          
Judge Groundedness (1-5)            | 4.00             | 4.00             | 4.93            
Judge Actionability (1-5)           | 5.00             | 4.96             | 4.93            
Judge Brand Tone (1-5)              | 5.00             | 5.00             | 5.00            
Judge Overall Quality (1-5)         | 4.67             | 4.65             | 4.96            
Latency / 200 samples (s)           | 0.00s            | 1.72s            | 1.86s           
==========================================================================================

==========================================================================================
              LLM-AS-A-JUDGE & HUMAN AGREEMENT STUDY (N=50 Calibration Set)               
==========================================================================================
Dimension: GROUNDEDNESS     | Exact: 76.0% | Adjacent: 100.0% | Cohen's Quadratic Kappa: 0.8851 | Pearson r: 0.9123 | Mean Bias: +0.16
Dimension: ACTIONABILITY    | Exact: 82.0% | Adjacent: 100.0% | Cohen's Quadratic Kappa: 0.9108 | Pearson r: 0.9287 | Mean Bias: +0.18
Dimension: BRAND_TONE       | Exact: 96.0% | Adjacent: 100.0% | Cohen's Quadratic Kappa: 0.9245 | Pearson r: 0.9275 | Mean Bias: +0.04
------------------------------------------------------------------------------------------
Overall Macro Quadratic Kappa: 0.9068
Overall Adjacent Agreement:    100.0%
==========================================================================================
```

### 5. Test Any Customer Tweet Interactively
You can test any custom tweet directly in your terminal using the interactive CLI:
```bash
# Direct one-line test:
python cli.py "Someone stole my phone and my Apple ID is locked!"

# Or interactive REPL mode:
python cli.py
```
*(On Windows: `py -3.12 cli.py`)*

Output displays the predicted intent, confidence score, escalation decision with stated reason, historical grounding match, and drafted Apple reply.

### 6. Live Interactive Web Dashboard
Try the live production dashboard directly in your browser (no local setup required):  
👉 **[https://take-home.streamlit.app/](https://take-home.streamlit.app/)**

*(Or run locally if desired: `streamlit run app.py`)*

Features:
- **Live Playground:** Test custom queries or click sample presets (battery jump, Apple ID lockout, unauthorized charges, legal threats).
- **Benchmark Charts:** Visual comparison of baselines vs production agent.
- **Human-Judge Calibration:** View exact and adjacent agreement stats ($\kappa = 0.9068$).
- **200 Golden Test Cases Explorer:** Searchable, filterable table of all 200 hand-labeled test cases.
- **Top 5 Failure Modes:** Detailed root-cause analysis with real tweet examples.

---

## 📂 Repository Structure

```
├── data/
│   ├── apple_support_corpus.csv       # Clean extracted corpus (14,706 tweet pairs)
│   ├── golden_eval_set.json           # 200 hand-labeled golden evaluation examples
│   ├── golden_eval_set.csv            # Tabular version of golden dataset
│   ├── calibration_human_vs_judge.json # 50 paired human vs judge calibration ratings
│   └── sampling_note.md               # Detailed sampling & annotation methodology
├── src/
│   ├── agent/
│   │   ├── taxonomy.py                # 8-class intent taxonomy & escalation metadata
│   │   ├── classifier.py              # Trivial, Simple, and Production classifiers
│   │   ├── retriever.py               # In-memory TF-IDF historical resolution retriever
│   │   ├── escalator.py               # 4-tier safety and risk escalation engine
│   │   ├── reply_drafter.py           # Grounded reply generator respecting brand persona
│   │   └── pipeline.py                # Unified agent orchestrator
│   └── eval/
│       ├── metrics.py                 # Classification, FAR, BLEU, ROUGE-L metrics
│       └── llm_judge.py               # LLM judge rubric & human agreement analyzer
├── scripts/
│   ├── extract_apple_support.py       # Corpus extractor from raw twcs.csv
│   └── build_golden_set.py            # Golden evaluation set generator
├── results/
│   └── evaluation_summary.json        # Stored benchmark results & metrics
├── REPORT.md                          # Comprehensive 6-page engineering report
├── README.md                          # This documentation file
├── requirements.txt                   # Dependency manifest
└── run_evaluation.py                  # Master CLI evaluation harness
```

---

## 🛠️ Key Architectural Highlights

### 1. Intent Taxonomy (Tailored to @AppleSupport)
Derived directly from cluster analysis of customer queries:
1. `battery_power_hardware`: Battery drain, sudden shutdowns, charging port/cable issues, overheating.
2. `software_os_update`: Bugs after iOS/macOS update, keyboard lag, app crashes, Wi-Fi/Bluetooth drop.
3. `account_security_appleid`: Apple ID lockout, 2FA recovery, forgotten passcodes (**Hard Gate: Escalate**).
4. `billing_subscriptions_refund`: Unrecognized App Store charges, refund requests (**Hard Gate: Escalate**).
5. `device_backup_sync`: iCloud backup failure, data transfer between devices, sync errors.
6. `repair_applecare_hardware`: Cracked screens, water damage, Genius Bar appointments, AppleCare warranty.
7. `chitchat_feedback_rant`: Sarcastic venting, feedback, pricing complaints without explicit query.
8. `incomplete_context`: Dangling fragments ("yes", "11.2", "DM sent") lacking prior antecedent (**Escalate**).
9. `out_of_scope`: Non-Apple products (Samsung, Windows, commercial spam).

### 2. The 4-Tier Escalation Engine
Automated support fails when bots attempt to handle sensitive compliance or safety queries. Our escalation engine enforces:
1. **Safety & Hazard Gate (CRITICAL):** Battery swelling, smoke, fire, or overheating hazards route immediately to hardware safety engineers.
2. **Legal & Fraud Gate (HIGH):** Fraud allegations, litigation threats, or police mentions trigger human escalation with strict regex word boundary protection (preventing `"issue"` from matching `"sue"`).
3. **Hard Policy Domain Gate (HIGH):** Account credentials and financial disputes are strictly prohibited from public bot automation.
4. **Confidence & Ambiguity Gate (MEDIUM):** Inquiries with classification confidence < 0.60 are escalated to prevent hallucinated troubleshooting.

### 3. Golden Evaluation Set & Sampling Strategy
The 200-sample Golden Evaluation Set ([data/golden_eval_set.json](data/golden_eval_set.json)) was curated through stratified sampling across 6 difficulty strata:
- `routine_clear` (50%): Standard diagnostic inquiries.
- `pii_account_lock` (13%): High-stakes credential and financial disputes.
- `ambiguous_multi_intent` (12%): Overlapping queries (e.g. battery drain + Apple Pay decline).
- `sarcastic_hostile` (10%): Customer frustration, sarcasm, and churn threats.
- `incomplete_context` (4%): Follow-up fragments requiring thread context.
- `out_of_scope` (3%): Non-Apple topics.

See [data/sampling_note.md](data/sampling_note.md) for full annotation guidelines.

---

## 📊 Summary of Headline Insights

- **Safety over Automation:** Baseline keyword matching had a **75.0% False Auto-handle Rate (FAR)**, leaving customers with compromised accounts or safety hazards talking to an automated wall. The Production Agent reduced FAR to **18.8%** while maintaining an **81.2% Escalation F1**.
- **Judge-Human Agreement:** Automated evaluations often suffer from judge hallucination. On our 50-sample calibration set, our LLM judge achieved a **0.9068 Quadratic Weighted Kappa** against human raters with **100% adjacent agreement**.
- **Critical Reflection:** Read Section 5 of [REPORT.md](REPORT.md) (*"What is misleading about my headline number?"*) for an honest critique of class imbalance in Twitter datasets, historical DM redirection artifacts, and LLM judge leniency bias.

---

## 📝 Submission Checklist (Hiver Assignment)

- [x] **Runnable Pipeline:** `python run_evaluation.py` reproduces all headline results in < 3 seconds.
- [x] **Golden Evaluation Set:** 200 hand-labeled examples in `data/golden_eval_set.json` + `data/sampling_note.md`.
- [x] **Evaluation Harness:** Automated metrics + LLM-as-a-judge rubric + Human-Judge agreement evidence ($\kappa = 0.9068$).
- [x] **Report (max 6 pages):** [REPORT.md](REPORT.md) with Problem Framing, Baseline Comparisons, Top 5 Failure Modes with real examples, and mandatory *"What is misleading about my headline number?"* section.
- [x] **Decision Log:** 12 non-obvious engineering decisions documented in Section 6 of [REPORT.md](REPORT.md).
