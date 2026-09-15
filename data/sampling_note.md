# Golden Evaluation Set — Sampling & Annotation Methodology

## 1. Overview
The Golden Evaluation Set consists of **200 hand-labeled examples** drawn from real customer tweets directed to `@AppleSupport` (from the Kaggle Customer Support on Twitter dataset) combined with targeted real-world adversarial and edge-case samples.

## 2. Intent Taxonomy
We defined an 8-class taxonomy tailored specifically to Apple customer service operations:
1. `software_os_update` (5 examples): iOS/macOS update glitches, keyboard lag, app crashes, connectivity bugs.
2. `battery_power_hardware` (4 examples): Battery drain, sudden percentage drops, charging port issues, overheating.
3. `account_security_appleid` (4 examples): Locked Apple ID, 2FA recovery, forgotten passcodes, stolen devices.
4. `billing_subscriptions_refund` (6 examples): App Store unauthorized charges, refund requests, subscription cancellations.
5. `device_backup_sync` (166 examples): iCloud sync failures, device-to-device migration, WhatsApp/photo backups.
6. `repair_applecare_hardware` (5 examples): Cracked screens, Genius Bar appointments, hardware replacements, AppleCare warranty claims.
7. `chitchat_feedback_rant` (3 examples): Customer venting, sarcastic remarks, brand feedback without direct actionable queries.
8. `out_of_scope` (3 examples): Non-Apple queries (Android, Windows, pizza, unsolicited links).

## 3. Stratified Sampling Strategy
Customer queries on Twitter are non-uniform and messy. A simple random sample heavily over-indexes on generic "battery drains fast" tweets and misses high-risk edge cases. We stratified the 200 instances across 6 difficulty strata:

| Stratum | Count | Description |
| :--- | :--- | :--- |
| `routine_clear` | 174 | Straightforward troubleshooting or feature queries. |
| `pii_account_lock` | 8 | High-stakes security, credential lockout, or financial charge disputes. |
| `ambiguous_multi_intent` | 6 | Multi-issue inquiries (e.g. battery drain + Apple Pay decline). |
| `sarcastic_hostile` | 5 | Frustrated sentiment, sarcastic praise, or churn/legal threats. |
| `incomplete_context` | 4 | Follow-up fragments ("yes it is", "11.2", "DM sent") lacking prior context. |
| `out_of_scope` | 3 | Queries completely outside Apple Support domain. |

## 4. Escalation Policy & Annotation Guidelines
Each example is labeled with a binary `should_escalate` flag (16 Escalate, 184 Auto-Handle) and a concrete policy rationale:
- **Mandatory Escalation Criteria:**
  - **Security & PII:** Any request requiring password reset, Apple ID unlocking, 2FA bypass, or device activation lock removal. Automated handling of credentials violates basic security compliance.
  - **Financial Transactions:** Unrecognized credit card charges, refund disputes over $20, or duplicate billing.
  - **Severe Risk / Churn:** Threats of litigation, allegations of fraud, or hardware physical hazards (e.g. swollen battery, smoking charger).
  - **Context Missing:** Dangling single-word or pronoun tweets where automated responses would loop or frustrate the customer.
- **Auto-Handling Criteria:**
  - Standard technical troubleshooting where public diagnostic steps exist (e.g., reset network settings, check battery health, force restart).
  - Self-service instructional links (how to manage subscriptions, how to use Quick Start).
  - Polite de-escalation for non-actionable complaints and out-of-scope queries.
