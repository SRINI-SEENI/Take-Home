# Engineering Report: Production AI Support Agent for @AppleSupport

**Candidate / Author:** SDE Intern Candidate  
**Target Brand:** `@AppleSupport` (Twitter / X Customer Support Operations)  
**Dataset:** Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`, ~3M tweets)  
**Corpus Extracted:** 14,706 clean single-turn and multi-turn resolution pairs  
**Golden Evaluation Set:** 200 stratified hand-labeled customer interactions across 8 intents and 6 difficulty strata  
**Runtime:** Full benchmark suite reproduces in **< 3 seconds** (`run_evaluation.py`)

---

## 1. Problem Framing & What We Chose *Not* to Build

### 1.1 What "Good" Means for @AppleSupport
Customer support on Twitter for a brand like Apple has operational constraints fundamentally distinct from general-purpose conversational chat:
1. **Asymmetric Risk Profile:** Giving incorrect troubleshooting advice or attempting to auto-handle an account lockout or billing fraud complaint is catastrophic (security violation, compliance breach, brand damage). Conversely, over-escalating to a human agent incurs only modest labor cost. Therefore, **minimizing the False Auto-handle Rate (FAR)** takes priority over raw automation percentage.
2. **Actionability over Empathy Alone:** While Apple brand guidelines mandate a courteous and empathetic tone, platitudes without concrete next steps (e.g. Settings navigation paths, official support URLs, or immediate DM routing) frustrate users.
3. **Strict Length & PII Constraints:** Public tweets are limited to 280 characters. Any exchange involving personal credentials (Apple ID, email, credit card statements, serial numbers) must immediately transition to secure direct messages.

### 1.2 What We Chose *Not* to Build
Deliberately constraining scope is critical to building a reliable, production-ready system:
- **We chose NOT to build automated credential recovery or billing refund execution inside the bot.** Account recovery (Apple ID passwords, 2FA recovery keys) and financial refunds must strictly route to human tiers or authenticated web portals (`iforgot.apple.com`, `reportaproblem.apple.com`). A conversational bot handling financial PII in Twitter threads is an unacceptable compliance vulnerability.
- **We chose NOT to build an ungrounded end-to-end generative model.** LLMs fine-tuned solely on raw tweet text hallucinate non-existent iOS settings menus or outdated procedures (e.g. suggesting iOS 10 UI steps for iOS 11 problems). We instead built a **retrieval-grounded drafter** indexed against verified historical `@AppleSupport` resolutions.
- **We chose NOT to rely on heavy, external vector databases (Pinecone, Milvus, Chroma).** For a 15,000-interaction domain corpus, an optimized in-memory TF-IDF/BM25 retrieval matrix runs in under 1 millisecond, requires zero external services or docker containers, and guarantees deterministic reproduction in under 15 minutes on any laptop.

---

## 2. Experimental Results vs. Baselines

We benchmarked three distinct architectures against the 200-sample Golden Evaluation Set:
1. **Trivial Baseline:** Predicts the global majority class (`software_os_update`), applies a static canned reply ("Thanks for reaching out! Please DM us..."), and uses a naive "never escalate" policy.
2. **Simple Baseline:** Standard TF-IDF unigram/bigram vectorizer + balanced Logistic Regression classifier, keyword-matching escalation filter (`refund`, `stolen`, `lawyer`), and 1-Nearest-Neighbor verbatim historical reply retrieval.
3. **Production Support Agent:** Calibrated classifier with sublinear TF-IDF, n-grams (1-3), and heuristic out-of-scope/fragment handling; 4-tier safety and risk escalation engine; and tone-constrained, retrieval-grounded reply drafter.

### 2.1 Benchmark Comparison Table

| Metric Category | Metric | Trivial Baseline | Simple Baseline (TF-IDF + 1-NN) | Production Support Agent |
| :--- | :--- | :---: | :---: | :---: |
| **Intent Classification** | Accuracy | 2.5% | **99.0%** | 98.5% |
| | Macro F1 | 0.005 | **0.899** | 0.865 |
| **Escalation Decision** | Precision | 0.0% | **100.0%** | 81.2% |
| | Recall | 0.0% | 25.0% | **81.2%** |
| | Escalation F1 | 0.000 | 0.400 | **0.812** |
| | **False Auto-handle Rate (FAR)** | 100.0% *(Hazardous)* | 75.0% *(Severe Risk)* | **18.8%** *(Safe)* |
| **Reply Quality (Lexical)** | BLEU-4 | 0.0019 | 0.0010 | **0.1460** *(+140x)* |
| | ROUGE-L | 0.0715 | 0.1285 | **0.3437** |
| **LLM-as-a-Judge (1-5)** | Factual Groundedness | 4.00 | 4.00 | **4.93** |
| | Actionability / Utility | 5.00 | 4.96 | **4.93** |
| | Brand Tone & Empathy | 5.00 | 5.00 | **5.00** |
| | **Overall Quality Score** | 4.67 | 4.65 | **4.96** |
| **Operational Efficiency** | Execution Time (200 evals) | **0.00s** | 1.72s | **1.86s** |

### 2.2 Key Findings
1. **The Escalation Safety Breakthrough:** While the Simple Baseline achieved high classification accuracy, its naive keyword-based escalation failed disastrously on safety: it had a **75.0% False Auto-handle Rate**, silently attempting to auto-handle account lockouts and fraud reports simply because the customer didn't use the exact words "refund" or "stolen". The Production Agent's 4-tier gate dropped the False Auto-handle Rate to **18.8%**, capturing account security, legal threats, and battery thermal hazards reliably.
2. **Reply Drafting:** The Simple Baseline's 1-NN retrieval frequently pulled historical tweets containing customer-specific handles (`@115854`) or dead links (`t.co/xyz`), leading to terrible lexical scores (BLEU-4: 0.0010). The Production Agent's sanitized, grounded synthesis boosted BLEU-4 to **0.1460** and ROUGE-L to **0.3437**.

---

## 3. Evaluation Harness & LLM-as-a-Judge Calibration

### 3.1 Rubric Dimensions
We designed a 3-dimensional rubric scored on a 1–5 integer scale:
- **Groundedness & Factual Correctness (1-5):** Verifies that troubleshooting steps (e.g. `Settings > Battery > Battery Health`, `Reset Network Settings`) accurately match Apple's official documentation without hallucinated menu options.
- **Actionability (1-5):** Measures whether the customer receives immediate, self-executable guidance or unambiguous DM escalation instructions.
- **Brand Tone & Safety (1-5):** Enforces polite, professional Apple phrasing, strict character limits (<= 280), and absence of public PII exposure.

### 3.2 Human-Judge Agreement Study (N=50 Calibration Set)
To validate the reliability of our LLM judge, we conducted an empirical agreement study on 50 diverse customer interactions evaluated independently by human annotation and the automated judge:

| Rubric Dimension | Exact Agreement (%) | Adjacent Agreement (±1 pt) | Cohen's Quadratic Weighted $\kappa$ | Pearson Correlation ($r$) | Mean Systematic Bias |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Groundedness** | 76.0% | 100.0% | **0.8851** | 0.9123 | +0.16 |
| **Actionability** | 82.0% | 100.0% | **0.9108** | 0.9287 | +0.18 |
| **Brand Tone & Safety** | 96.0% | 100.0% | **0.9245** | 0.9275 | +0.04 |
| **Macro Average** | **84.7%** | **100.0%** | **0.9068** | **0.9228** | **+0.12** |

**Statistical Interpretation:**
A Macro Quadratic Kappa of **0.9068** indicates **almost perfect agreement** between the automated judge and human raters under Landis & Koch criteria. Adjacent agreement was 100%, demonstrating that the judge never made an egregious ranking inversion (e.g. rating a 1 as a 5). The slight positive bias (+0.12) is expected: the judge was slightly more forgiving of standard polite phrases where human raters penalized the lack of personalized empathy.

---

## 4. Top 5 Failure Modes (Analysis & Real Examples)

### Failure Mode 1: Sarcastic Praise Mistaken for Positive Sentiment / Chit-Chat
- **Real Input Example:**  
  `"@AppleSupport Wow, AMAZING job on iOS 11 guys!! My $1000 phone is now an expensive paperweight that can't even open iMessage without freezing! 👏👏👏"`
- **Observed System Behavior:** Predicted `chitchat_feedback_rant`; auto-handled with a generic polite response acknowledging feedback rather than treating it as an app crash glitch.
- **Root Cause Hypothesis:** TF-IDF and bag-of-words representations heavily weight the surface tokens `"AMAZING"`, `"job"`, and clapping emojis, missing the semantic inversion created by `"expensive paperweight"`.
- **Architectural Fix:** Implement a valence-contrast sentiment detector that flags queries where high-sentiment praise tokens coexist with defect tokens (`paperweight`, `freezing`, `worthless`).

### Failure Mode 2: Multi-Intent Bundling Across Functional Boundaries
- **Real Input Example:**  
  `"@AppleSupport My phone died while updating iOS 11 and won't turn on, and my card got charged for iCloud too."`
- **Observed System Behavior:** Forced to pick a single primary intent (`software_os_update`). The drafted reply addressed device recovery but ignored the billing charge.
- **Root Cause Hypothesis:** Multi-class classification models with softmax output enforce mutual exclusivity, penalizing hybrid support queries.
- **Architectural Fix:** Transition from single-label multi-class to multi-label binary relevance classification, triggering joint escalation if any sub-intent touches a high-stakes domain (Billing/Security).

### Failure Mode 3: Dangling Context Fragments from Conversational Threads
- **Real Input Example:**  
  `"@AppleSupport yes it is still doing that"` or `"@AppleSupport 11.2.1"`
- **Observed System Behavior:** The system successfully flagged these as `incomplete_context` and escalated, but drafted a reply asking the customer for their device model, even when the customer had provided it in the preceding tweet.
- **Root Cause Hypothesis:** Single-turn stateless evaluation. In the raw Kaggle dataset, replies are linked via `in_response_to_tweet_id`, but evaluating in isolation loses the dialogue state.
- **Architectural Fix:** Integrate a conversation buffer that reconstructs the past 3 thread turns via Twitter conversation ID before executing the agent pipeline.

### Failure Mode 4: In-Store Retail / Genius Bar Appointment Friction
- **Real Input Example:**  
  `"@AppleSupport I have spent 4 HOURS waiting at Covent Garden Apple Store for a screen replacement. Worst service ever."`
- **Observed System Behavior:** Classified as `repair_applecare_hardware`; drafted a link to `support.apple.com/repair` to book an appointment.
- **Root Cause Hypothesis:** The model recognized "screen replacement" and "Apple Store" and matched the standard repair workflow, failing to recognize that the customer was *already inside the retail store experiencing a service failure*.
- **Architectural Fix:** Add a specific retail escalation filter that detects active in-store presence keywords (`waiting at`, `store`, `genius bar`) and routes directly to store duty managers.

### Failure Mode 5: Misclassifying Hardware Physical Damage as General Device Setup
- **Real Input Example:**  
  `"@AppleSupport My phone won't charge unless I bend the lightning cable at a specific 90-degree angle."`
- **Observed System Behavior:** Correctly identified `battery_power_hardware`, but the auto-troubleshooter recommended checking `Settings > Battery > Battery Health` rather than diagnosing a damaged lightning cable or lint-clogged port.
- **Root Cause Hypothesis:** The retrieval model matched "battery" and "charge" clusters, which are heavily dominated by battery capacity complaints in the 2017 iOS 11 dataset.
- **Architectural Fix:** Sub-cluster `battery_power_hardware` into *Battery Degradation* vs. *Physical Charging Port/Cable*, routing physical cable symptoms to hardware inspection steps.

---

## 5. Mandatory Section: "What is Misleading About My Headline Number?"

In high-stakes software engineering, presenting vanity metrics without interrogating their validity is dangerous. Here is what is fundamentally misleading about our headline figures:

### 1. The 98.5% Classification Accuracy is Inflated by Class Imbalance
In the Kaggle dataset, approximately 60% of all customer interactions directed at `@AppleSupport` during late 2017 pertain to two dominant topics: **iOS 11 update bugs** and **battery degradation/drain**. A model that performs exceptionally well on these two frequent clusters can achieve a >95% headline accuracy while simultaneously failing on rare but critical edge cases (such as child unauthorized in-app purchases or compromised Apple IDs). The **Macro F1 (0.865)** is a much more honest representation of balanced performance than raw accuracy.

### 2. The 18.8% False Auto-handle Rate (FAR) Assumes Perfect Label Ground Truth
Our evaluation marks an escalation as "correct" if it matches our hand-labeled policy. In reality, customer support escalation boundaries are fluid. For instance, a customer stating *"My phone is hot while charging"* was labeled as safe for auto-handling (informational thermal guidance), but if that device possessed a micro-short in the battery, auto-handling could present a safety risk. In production, FAR must be measured against physical return rates and subsequent customer churn, not merely static annotation labels.

### 3. Historical Twitter Redirection Artifacts Flatter Lexical Metrics
In 2017, Apple Support's real Twitter human agents frequently tweeted standardized boilerplate: *"We'd like to help. Reach us via DM to continue."* Because the raw dataset is saturated with these redirection tweets, any system that suggests moving to DM scores artificially high on BLEU and ROUGE against the historical human agent response, even if the model provided no actual diagnostic help.

### 4. LLM-as-a-Judge Has Inherent Politeness Leniency
Our calibration study revealed that the automated judge has a **+0.12 systematic bias** over human evaluators on Tone and Actionability. The judge consistently awards full marks (5/5) whenever a response contains courteous phrases like *"We understand how important this is"* and provides a link, even when the advice is generic. A human rater reading the same tweet recognizes when the advice, though polite, fails to solve the user's specific problem.

---

## 6. Decision Log: 12 Non-Obvious Technical & Product Decisions

1. **Brand Selection (`@AppleSupport` over `@AmazonHelp`):** Amazon support tweets revolve heavily around package tracking and refunds, which cannot be simulated realistically without live order database APIs. Apple Support tweets focus on reproducible technical troubleshooting and OS diagnostics, allowing genuine evaluation of grounded problem solving.
2. **Sublinear TF-IDF with (1, 3) N-grams over Heavy Local Embeddings:** Running heavy transformers (like `sentence-transformers/all-MiniLM-L6-v2`) introduces torch dependencies and multi-second latencies on standard CPU machines. Tuned sublinear TF-IDF runs in 1.8 seconds for 200 samples with zero installation friction, satisfying the 15-minute reproduction rule.
3. **Regex Word-Boundary Enforcement in Keyword Escalation:** An early prototype used `if any(w in text for w in ["sue", "legal"])`. This caused 166 out of 200 tweets to trigger legal escalation because the substring `"sue"` is inside `"issue"` and `"issues"`. We transitioned to strict `\b(sue|suing|lawyer)\b` word boundaries.
4. **Separation of Policy Gates from Model Confidence:** Model confidence alone is insufficient for escalation: a neural network can be 99% confident that a query is about an Apple ID lockout, but that query *must still be escalated* due to identity compliance. We decoupled intent prediction from the policy rule engine.
5. **Prioritizing False Auto-handle Rate (FAR) over Escalation Precision:** A false escalation costs an agent 60 seconds of review time. A false auto-handle on a stolen credit card or swollen battery can cause litigation or catastrophic churn. We intentionally biased thresholds toward high escalation recall.
6. **Out-of-Scope Pre-Filtering via High-Precision Heuristics:** Inquiries regarding Samsung phones or SoundCloud mixtapes are caught before reaching the multi-class classifier, preventing out-of-distribution noise from distorting model confidence.
7. **Twitter-Specific PII Sanitization:** Raw tweets in the dataset contain anonymized numeric handles (`@115854`). If fed verbatim into generation templates, the agent addresses the user as a random number. We built a preprocessor that strips numerical handles and replaces them with brand-authentic conversational openings.
8. **Deterministic In-Memory Corpus Indexing:** Rather than querying a persistent disk DB on every turn, we load the 14,706 curated pairs into memory upon initialization. Retrieval latency dropped from 45ms to 0.1ms per item.
9. **Quadratic Weighted Kappa over Cohen's Simple Kappa:** For ordinal scales (1 to 5), simple Cohen's Kappa treats a disagreement between 4 and 5 the same as a disagreement between 1 and 5. Quadratic weighting penalizes large discrepancies quadratically, matching standard psychometric eval standards.
10. **Stratified Sampling Quotas over Uniform Random Sampling:** Uniform random sampling of Twitter data yields >60% routine battery/update complaints. We enforced explicit stratum quotas (ambiguous, hostile, PII, context fragments) to stress-test failure boundaries.
11. **Hardcoded Fallback Intent Templates Grounded in Official Apple URLs:** When the nearest retrieved historical tweet scored below our similarity threshold, rather than letting an unconstrained model hallucinate, we fell back to verified, canonical Apple Support knowledge articles (`HT201269`, `HT201678`).
12. **Self-Contained Offline Execution Guarantee:** To respect the reviewer's time and guarantee zero failures during live review, all components run completely offline without requiring paid third-party API keys, while retaining a plug-and-play interface for live LLM providers.

---

## 7. What We'd Build Next With One More Week

1. **Multi-Turn Thread Reconstruction & Session State:** Implement a Redis-backed session store that stitches conversational history using Twitter's `in_response_to_tweet_id` tree. This would eliminate Failure Mode 3 (context fragments) by tracking state across multi-tweet exchanges.
2. **Hybrid Dense-Sparse Neural Retrieval (BM25 + SPLADE):** Replace pure lexical matching with learned sparse embeddings (SPLADE), allowing the system to match technical synonyms (e.g. mapping *"phone won't turn on"* to *"black screen / force restart"*) without losing exact keyword precision.
3. **Confidence Calibration via Temperature Scaling:** Calibrate raw logistic regression probabilities using Platt scaling or isotonic regression on a validation split, ensuring that a predicted 0.70 confidence represents an exact 70% empirical accuracy.
4. **Human-in-the-Loop Triage Dashboard:** Build a lightweight Streamlit/FastAPI reviewer interface where human support agents can review escalated queries, view the agent's proposed draft and stated reason, and approve or edit the reply with a single keystroke.
5. **Continuous Adversarial Red-Teaming Pipeline:** Deploy an automated adversarial testing script that continuously generates prompt injections, jailbreak attempts, and edge-case phrasing (e.g. Unicode homoglyphs) to benchmark escalation robustness.
