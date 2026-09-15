"""
app.py
Production AI Customer Support Agent & Evaluation Dashboard for @AppleSupport.
Features:
- Apple Command Center UI with high-contrast, theme-resilient styling
- Live Interactive Playground (test any custom query or select realistic edge cases)
- Real-time Intent Classification, Grounded Retrieval, and 4-Tier Escalation Engine
- Benchmark Comparison Table & Interactive Visualizations
- LLM-as-a-Judge & Human Calibration Explorer (Cohen's Kappa = 0.9068)
- Searchable Explorer for all 200 Golden Test Cases
- Deep Dive into Top 5 Real-World Failure Modes
"""

import os
import json
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="@AppleSupport AI Command Center",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast CSS (Guaranteed crystal-clear readability in both Dark & Light themes)
st.markdown("""
<style>
    /* Global Container Styles */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 24px;
        border-radius: 12px;
        color: #ffffff !important;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .app-header h1 {
        color: #ffffff !important;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
    }
    .app-header p {
        color: #cbd5e1 !important;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* High-contrast Card Boxes */
    .section-card {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Customer Tweet Box */
    .customer-tweet-box {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-left: 5px solid #2563eb !important;
        border-radius: 8px;
        padding: 14px;
        font-size: 1rem;
        line-height: 1.5;
    }
    .customer-tweet-box * {
        color: #0f172a !important;
    }

    /* Historical Knowledge Box */
    .history-box {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-left: 5px solid #64748b !important;
        border-radius: 8px;
        padding: 14px;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .history-box * {
        color: #0f172a !important;
    }

    /* Brand Drafted Reply Box */
    .reply-box {
        background-color: #f0fdf4 !important;
        color: #064e3b !important;
        border: 1px solid #86efac !important;
        border-left: 5px solid #16a34a !important;
        border-radius: 8px;
        padding: 16px;
        font-size: 1.02rem;
        font-weight: 500;
        line-height: 1.5;
    }
    .reply-box * {
        color: #064e3b !important;
    }

    /* Badges */
    .badge-auto {
        background-color: #16a34a !important;
        color: #ffffff !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-escalate {
        background-color: #dc2626 !important;
        color: #ffffff !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-intent {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Lazy-loaded agent pipeline cache
@st.cache_resource(show_spinner="Initializing @AppleSupport AI Agent Pipeline...")
def load_agent(version: str = "v5_dynamic_cable_port_patch_2026_09_15"):
    from run_evaluation import load_data, prepare_train_labels
    from src.agent.pipeline import SupportAgentPipeline
    golden_data, train_df = load_data()
    train_texts, train_labels = prepare_train_labels(train_df, golden_data)
    agent = SupportAgentPipeline(mode="production")
    agent.train_classifier(train_texts, train_labels)
    agent.retriever.load_and_index()
    return agent

@st.cache_data
def load_eval_data():
    summary_path = os.path.join("results", "evaluation_summary.json")
    pred_path = os.path.join("results", "detailed_golden_predictions.csv")
    summary = {}
    detailed_df = pd.DataFrame()
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    if os.path.exists(pred_path):
        detailed_df = pd.read_csv(pred_path)
    return summary, detailed_df

agent = load_agent()
summary_data, detailed_df = load_eval_data()

# Helper function to compute result with fresh ReplyDrafter
def compute_agent_result(text_to_process: str):
    from src.agent.reply_drafter import ReplyDrafter
    from src.agent.escalator import EscalationDecision
    res_pipeline = agent.process_message(text_to_process)
    fresh_drafter = ReplyDrafter()
    esc = EscalationDecision(
        should_escalate=res_pipeline.should_escalate,
        reason=res_pipeline.escalation_reason,
        risk_level=res_pipeline.risk_level,
        trigger_rule=""
    )
    res_pipeline.drafted_reply = fresh_drafter.draft(
        customer_text=text_to_process,
        predicted_intent=res_pipeline.intent,
        escalation=esc,
        historical_resolutions=res_pipeline.retrieved_resolutions
    )
    return res_pipeline

# Sidebar
with st.sidebar:
    st.markdown("### 🍎 **@AppleSupport AI**")
    st.caption("Production Operations & Triage Agent")
    st.markdown("---")
    st.markdown("**Corpus Grounding**")
    st.markdown("• **14,706** Historical Verified Pairs")
    st.markdown("• **200** Hand-labeled Test Cases")
    st.markdown("• **Cohen's Kappa:** `0.9068`")
    st.markdown("---")
    st.markdown("**Quick Links**")
    st.markdown("[📄 Read Full REPORT.md](https://github.com/SRINI-SEENI/Take-Home/blob/main/REPORT.md)")
    st.markdown("[🐙 GitHub Repository](https://github.com/SRINI-SEENI/Take-Home)")
    st.markdown("---")
    st.info("Tip: Run terminal benchmarks with:\n`py -3.12 run_evaluation.py`")

# Header Banner
st.markdown("""
<div class="app-header">
    <h1>🍎 @AppleSupport AI Operations & Evaluation Dashboard</h1>
    <p>Grounded customer support automation, 4-tier risk escalation, and empirical human-judge calibration.</p>
</div>
""", unsafe_allow_html=True)

# Tab Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Live Support Playground",
    "📊 Benchmark vs Baselines",
    "⚖️ LLM-as-a-Judge Calibration",
    "🔍 200 Golden Test Cases",
    "⚠️ Top 5 Failure Modes"
])

# ---------------------------------------------------------
# TAB 1: LIVE PLAYGROUND
# ---------------------------------------------------------
with tab1:
    st.markdown("### Test Customer Inquiries in Real Time")
    st.write("Type any custom customer tweet or click one of the realistic scenario presets below:")

    # Manage session state for tweet input so it never resets or overrides typed input
    if "tweet_input" not in st.session_state:
        st.session_state["tweet_input"] = "@AppleSupport My iPhone 7 battery percentage jumps from 40% to 1% in five minutes. Is this a battery degradation issue?"

    def set_tweet_query(text: str):
        st.session_state["tweet_input"] = text
        st.session_state["current_result"] = compute_agent_result(text)

    # Preset Quick Buttons
    preset_cols = st.columns(4)
    
    with preset_cols[0]:
        st.button("🔋 Battery Drain Jump", use_container_width=True, on_click=set_tweet_query, args=("@AppleSupport My iPhone 7 battery percentage jumps from 40% to 1% in five minutes. Is this a battery degradation issue?",))
        st.button("🔌 Bending Cable Angle", use_container_width=True, on_click=set_tweet_query, args=("@AppleSupport My phone won't charge unless I bend the lightning cable at a 90-degree angle. Is it cable or port?",))
            
    with preset_cols[1]:
        st.button("🔐 Stolen Phone / Locked ID", use_container_width=True, on_click=set_tweet_query, args=("@AppleSupport Someone stole my phone and changed my Apple ID trusted numbers! URGENT help!",))
        st.button("💳 Unauthorized iTunes Charge", use_container_width=True, on_click=set_tweet_query, args=("@AppleSupport I was charged $49.99 on iTunes that I never authorized! I need a refund immediately.",))

    with preset_cols[2]:
        st.button("😡 Sarcastic Hostile Rant", use_container_width=True, on_click=set_tweet_query, args=("@AppleSupport Wow AMAZING job on iOS 11 guys!! My $1000 phone is now an expensive paperweight 👏👏👏",))
        st.button("⚖️ Legal / Lawyer Threat", use_container_width=True, on_click=set_tweet_query, args=("@AppleSupport You stole money from my card and your staff hung up. Contacting my lawyer and suing you today.",))

    with preset_cols[3]:
        st.button("📲 Device Migration", use_container_width=True, on_click=set_tweet_query, args=("@AppleSupport How do I transfer all my contacts and photos from my old iPhone 6 to my new iPhone 8?",))
        st.button("🤖 Non-Apple Scope Question", use_container_width=True, on_click=set_tweet_query, args=("@AppleSupport Can you tell me how to root my Samsung Galaxy S8 to install a custom ROM?",))

    # Input text box using session state key
    user_tweet = st.text_area(
        "Incoming Customer Tweet:",
        key="tweet_input",
        height=85,
        placeholder="Type any custom tweet or question here..."
    )

    # If no result has been calculated yet, run once on default
    if "current_result" not in st.session_state:
        st.session_state["current_result"] = compute_agent_result(st.session_state["tweet_input"])

    if st.button("⚡ Run AI Agent Pipeline", type="primary", use_container_width=True):
        if user_tweet.strip():
            with st.spinner("Classifying intent, retrieving historical fixes, and evaluating safety..."):
                st.session_state["current_result"] = compute_agent_result(user_tweet)

    res = st.session_state.get("current_result")

    if res:
        st.markdown("---")

        # Two Column Display
        left_col, right_col = st.columns([1, 1], gap="medium")

        # Left Column: Intent & Escalation Decision
        with left_col:
            with st.container(border=True):
                st.markdown("#### 🎯 1. Intent Classification")
                st.markdown(f'<div class="badge-intent">{res.intent}</div>', unsafe_allow_html=True)
                st.write("")
                st.progress(min(1.0, float(res.confidence)))
                st.write(f"Confidence Score: **{res.confidence * 100:.1f}%**")

            with st.container(border=True):
                st.markdown("#### 🛡️ 2. Safety Escalation Engine")
                if res.should_escalate:
                    st.markdown('<div class="badge-escalate">🚨 ESCALATE TO HUMAN SPECIALIST</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-auto">✅ SAFE FOR AUTO-HANDLING</div>', unsafe_allow_html=True)
                
                st.write("")
                st.markdown(f"**Risk Severity:** `{res.risk_level}`")
                st.markdown(f"**Stated Policy Rationale:**")
                st.info(res.escalation_reason)

        # Right Column: Historical Grounding & Drafted Reply
        with right_col:
            with st.container(border=True):
                st.markdown("#### 📚 3. Grounded Historical Resolution")
                if res.retrieved_resolutions:
                    top_match = res.retrieved_resolutions[0]
                    sim = top_match.get("similarity_score", 0)
                    st.markdown(f"**Top Match from 14,706 Corpus (Similarity: `{sim:.2f}`):**")
                    st.markdown(f"""
                    <div class="history-box">
                        <strong>Past Customer Query:</strong><br>
                        "{top_match.get('historical_query', '')[:120]}..."<br><br>
                        <strong>Verified Official Fix:</strong><br>
                        "{top_match.get('historical_resolution', '')[:140]}..."
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Direct escalation routing applied (no automated fix recommended).")

            with st.container(border=True):
                st.markdown("#### 💬 4. Drafted Apple Support Reply")
                char_count = len(res.drafted_reply)
                st.markdown(f"""
                <div class="reply-box">
                    {res.drafted_reply}
                </div>
                """, unsafe_allow_html=True)
                
                if char_count <= 280:
                    st.success(f"✅ Length: **{char_count}/280 characters** (Strictly within Twitter limit)")
                else:
                    st.error(f"⚠️ Length: **{char_count}/280 characters** (Exceeds Twitter limit)")

# ---------------------------------------------------------
# TAB 2: BENCHMARK COMPARISON
# ---------------------------------------------------------
with tab2:
    st.markdown("### System Results vs. Two Baselines (N=200 Golden Set)")
    st.write("Empirical comparison against the Trivial Baseline (majority class) and Simple Baseline (TF-IDF + keywords):")

    # High level KPI cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Intent Accuracy", "98.5%", "+0.5% vs Simple")
    kpi2.metric("Escalation F1", "0.849", "+112% vs Simple")
    kpi3.metric("False Auto-handle Rate", "12.5%", "-83% Risk Reduction")
    kpi4.metric("Judge Overall Score", "4.95 / 5", "+0.30 Quality")

    st.write("")
    
    # Detailed Benchmark Table
    benchmark_df = pd.DataFrame([
        {"Metric Category": "Intent Classification", "Metric": "Accuracy", "Trivial Baseline": "2.5%", "Simple Baseline": "98.0%", "Production Agent": "98.5%"},
        {"Metric Category": "Intent Classification", "Metric": "Macro F1", "Trivial Baseline": "0.005", "Simple Baseline": "0.900", "Production Agent": "0.912"},
        {"Metric Category": "Escalation Decision", "Metric": "Precision", "Trivial Baseline": "0.0%", "Simple Baseline": "100.0%", "Production Agent": "82.3%"},
        {"Metric Category": "Escalation Decision", "Metric": "Recall", "Trivial Baseline": "0.0%", "Simple Baseline": "25.0%", "Production Agent": "87.5%"},
        {"Metric Category": "Escalation Decision", "Metric": "Escalation F1", "Trivial Baseline": "0.000", "Simple Baseline": "0.400", "Production Agent": "0.849"},
        {"Metric Category": "Escalation Decision", "Metric": "False Auto-handle Rate (FAR)", "Trivial Baseline": "100.0% (Fatal)", "Simple Baseline": "75.0% (Hazardous)", "Production Agent": "12.5% (Safe)"},
        {"Metric Category": "Reply Quality", "Metric": "BLEU-4", "Trivial Baseline": "0.0019", "Simple Baseline": "0.0010", "Production Agent": "0.1469 (+140x)"},
        {"Metric Category": "Reply Quality", "Metric": "ROUGE-L", "Trivial Baseline": "0.0715", "Simple Baseline": "0.1285", "Production Agent": "0.3453"},
        {"Metric Category": "LLM Judge (1-5)", "Metric": "Factual Groundedness", "Trivial Baseline": "4.00", "Simple Baseline": "4.00", "Production Agent": "4.93"},
        {"Metric Category": "LLM Judge (1-5)", "Metric": "Overall Quality", "Trivial Baseline": "4.67", "Simple Baseline": "4.65", "Production Agent": "4.95"},
        {"Metric Category": "Runtime", "Metric": "Latency / 200 evals", "Trivial Baseline": "0.00s", "Simple Baseline": "1.74s", "Production Agent": "1.68s"}
    ])
    st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

    st.markdown("#### Escalation Recall vs. False Auto-Handle Rate (FAR)")
    chart_df = pd.DataFrame({
        "Model": ["Trivial Baseline", "Simple Baseline (Keywords)", "Production Agent"],
        "Escalation Recall (%)": [0.0, 25.0, 87.5],
        "False Auto-handle Rate (%)": [100.0, 75.0, 12.5]
    })
    st.bar_chart(chart_df.set_index("Model"), height=280)

# ---------------------------------------------------------
# TAB 3: LLM-AS-A-JUDGE CALIBRATION
# ---------------------------------------------------------
with tab3:
    st.markdown("### LLM-as-a-Judge & Human Agreement Calibration Study (N=50)")
    st.write("Empirical validation proving our automated judge agrees with human expert annotators under psychometric standards:")

    col_a, col_b = st.columns([1, 1], gap="medium")

    with col_a:
        with st.container(border=True):
            st.markdown("#### Evaluation Rubric (1–5 Scale)")
            st.markdown("""
            - **1. Factual Groundedness (1-5):** Verifies that troubleshooting guidance (e.g. `Settings > Battery > Battery Health`) factually matches Apple documentation without hallucinated menus.
            - **2. Actionability (1-5):** Measures whether the reply provides concrete, self-executable guidance or clear DM instructions, rather than unhelpful canned phrases.
            - **3. Brand Tone & Safety (1-5):** Enforces courteous, empathetic Apple tone, strict character bounds (<= 280), and zero public PII disclosure.
            """)

    with col_b:
        with st.container(border=True):
            st.markdown("#### Inter-Annotator Agreement Statistics")
            agreement_df = pd.DataFrame([
                {"Dimension": "Groundedness", "Exact %": "76.0%", "Adjacent (±1 pt)": "100.0%", "Cohen's Quadratic Kappa": "0.8851", "Pearson r": "0.9123", "Mean Bias": "+0.16"},
                {"Dimension": "Actionability", "Exact %": "82.0%", "Adjacent (±1 pt)": "100.0%", "Cohen's Quadratic Kappa": "0.9108", "Pearson r": "0.9287", "Mean Bias": "+0.18"},
                {"Dimension": "Brand Tone", "Exact %": "96.0%", "Adjacent (±1 pt)": "100.0%", "Cohen's Quadratic Kappa": "0.9245", "Pearson r": "0.9275", "Mean Bias": "+0.04"},
                {"Dimension": "MACRO AVERAGE", "Exact %": "84.7%", "Adjacent (±1 pt)": "100.0%", "Cohen's Quadratic Kappa": "0.9068", "Pearson r": "0.9228", "Mean Bias": "+0.12"}
            ])
            st.dataframe(agreement_df, hide_index=True, use_container_width=True)
            st.success("✅ **Macro Quadratic Kappa of 0.9068** establishes *almost-perfect agreement* with zero ranking inversions (100% adjacent agreement).")

# ---------------------------------------------------------
# TAB 4: 200 GOLDEN TEST CASES EXPLORER
# ---------------------------------------------------------
with tab4:
    st.markdown("### 200 Hand-Labeled Golden Evaluation Set Explorer")
    st.write("Filter, search, and inspect the individual predictions across all 200 test cases:")

    if not detailed_df.empty:
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            strata_list = ["All"] + sorted(list(detailed_df["difficulty_stratum"].dropna().unique()))
            chosen_stratum = st.selectbox("Filter by Stratum:", strata_list)
        with f_col2:
            intent_list = ["All"] + sorted(list(detailed_df["gold_intent"].dropna().unique()))
            chosen_intent = st.selectbox("Filter by Intent:", intent_list)
        with f_col3:
            chosen_match = st.selectbox("Filter by Outcome:", ["All", "Only Correct", "Only Mismatches"])

        f_df = detailed_df.copy()
        if chosen_stratum != "All":
            f_df = f_df[f_df["difficulty_stratum"] == chosen_stratum]
        if chosen_intent != "All":
            f_df = f_df[f_df["gold_intent"] == chosen_intent]
        if chosen_match == "Only Correct":
            f_df = f_df[(f_df["intent_match"] == True) & (f_df["escalation_match"] == True)]
        elif chosen_match == "Only Mismatches":
            f_df = f_df[(f_df["intent_match"] == False) | (f_df["escalation_match"] == False)]

        st.caption(f"Showing **{len(f_df)}** of 200 cases:")
        st.dataframe(
            f_df[[
                "id", "difficulty_stratum", "customer_text", "gold_intent", "predicted_intent",
                "gold_escalate", "predicted_escalate", "drafted_reply", "judge_overall"
            ]],
            use_container_width=True,
            height=420
        )
    else:
        st.warning("Prediction data not found. Run `py -3.12 run_evaluation.py` to generate.")

# ---------------------------------------------------------
# TAB 5: TOP 5 FAILURE MODES
# ---------------------------------------------------------
with tab5:
    st.markdown("### Deep Dive: Top 5 Real Failure Modes")
    st.write("Honest engineering breakdown of real edge cases from Twitter data and their architectural remedies:")

    with st.expander("1. Sarcastic Praise Mistaken for Positive Sentiment"):
        st.markdown("""
        - **Real Tweet:** `"@AppleSupport Wow, AMAZING job on iOS 11 guys!! My $1000 phone is now an expensive paperweight that can't even open iMessage without freezing! 👏👏👏"`
        - **Observed Behavior:** Classified as `chitchat_feedback_rant`; auto-handled with polite thanks.
        - **Root Cause:** TF-IDF heavily weighted surface praise tokens (`AMAZING`, `job`, emojis), missing the semantic inversion caused by `expensive paperweight`.
        - **Architectural Fix:** Implement a valence-contrast sentiment detector that flags queries where high-sentiment tokens coexist with hardware defect tokens (`paperweight`, `freezing`).
        """)

    with st.expander("2. Multi-Intent Bundling Across Functional Boundaries"):
        st.markdown("""
        - **Real Tweet:** `"@AppleSupport My phone died while updating iOS 11 and won't turn on, and my card got charged for iCloud too."`
        - **Observed Behavior:** Forced into single intent (`software_os_update`). The reply addressed recovery but ignored the billing dispute.
        - **Root Cause:** Softmax multi-class output enforces mutual exclusivity on hybrid inquiries.
        - **Architectural Fix:** Transition to multi-label binary relevance classification, escalating if any sub-intent touches high-stakes domains (Billing/Security).
        """)

    with st.expander("3. Conversational Context Fragments"):
        st.markdown("""
        - **Real Tweet:** `"@AppleSupport yes it is still doing that"` or `"@AppleSupport 11.2.1"`
        - **Observed Behavior:** Successfully flagged as fragment and escalated, but drafted a reply asking for the iOS version the customer just gave.
        - **Root Cause:** Single-turn stateless evaluation loses conversational history.
        - **Architectural Fix:** Connect a Redis session buffer to stitch prior thread turns via Twitter's `in_response_to_tweet_id`.
        """)

    with st.expander("4. In-Store Retail / Genius Bar Friction"):
        st.markdown("""
        - **Real Tweet:** `"@AppleSupport I have spent 4 HOURS waiting at Covent Garden Apple Store for a screen replacement. Worst service ever."`
        - **Observed Behavior:** Classified as repair and offered an online appointment link.
        - **Root Cause:** Matched "screen replacement" and "store", unaware the customer was *already physically waiting inside the retail store*.
        - **Architectural Fix:** Retail escalation filter detecting in-store keywords (`waiting at`, `store`, `genius bar`) to alert store managers.
        """)

    with st.expander("5. Physical Cable Damage Confused with Battery Degradation"):
        st.markdown("""
        - **Real Tweet:** `"@AppleSupport My phone won't charge unless I bend the lightning cable at a specific 90-degree angle."`
        - **Observed Behavior:** Identified battery intent, but suggested checking `Settings > Battery > Battery Health`.
        - **Root Cause:** The retrieval corpus from 2017 is heavily dominated by battery capacity complaints.
        - **Architectural Fix:** Sub-cluster `battery_power_hardware` into *Battery Degradation* vs *Physical Port/Cable Damage*.
        """)

st.markdown("---")
st.caption("🍎 Built for Hiver SDE Intern Take-Home Assignment | Candidate Submission")
