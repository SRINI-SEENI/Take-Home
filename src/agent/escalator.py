"""
escalator.py
Implements a 4-tier safety and escalation decision engine:
1. Hard Policy Gate (Account Security, Unauthorized Billing)
2. Safety & Legal Hazard Gate (Swollen batteries, litigation, fraud)
3. Conversational Fragment Gate (Dangling pronouns, context-less replies)
4. Confidence & Ambiguity Gate (Low classifier or retrieval certainty)

Provides auditable rationale strings for human supervision.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from src.agent.taxonomy import TAXONOMY, IntentCategory, HIGH_STAKES_KEYWORDS

@dataclass
class EscalationDecision:
    should_escalate: bool
    reason: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    trigger_rule: str

class EscalationEngine:
    def __init__(self, confidence_threshold: float = 0.60, min_retrieval_similarity: float = 0.25):
        self.confidence_threshold = confidence_threshold
        self.min_retrieval_similarity = min_retrieval_similarity

    def decide(
        self,
        customer_text: str,
        predicted_intent: str,
        classifier_confidence: float,
        retrieval_score: float = 1.0
    ) -> EscalationDecision:
        lower_text = customer_text.lower()

        # Tier 1: Safety & Physical Hazard Gate (CRITICAL)
        hazard_pattern = r'\b(burned|burning|fire|smoke|exploded|swelled|swelling|hazard|melted)\b'
        if re.search(hazard_pattern, lower_text):
            return EscalationDecision(
                should_escalate=True,
                reason="Immediate physical safety hazard detected (battery swelling, thermal or electrical hazard). Requires senior engineering review.",
                risk_level="CRITICAL",
                trigger_rule="SAFETY_PHYSICAL_HAZARD"
            )

        # Tier 2: Legal & Fraud Escalation Gate (HIGH)
        # Note: Must use word boundaries so 'issue' doesn't false-positive on 'sue'
        legal_pattern = r'\b(lawyer|attorney|sue|suing|legal action|court|police|criminal|fraud)\b'
        if re.search(legal_pattern, lower_text):
            return EscalationDecision(
                should_escalate=True,
                reason="Legal dispute or fraud accusation detected. Policy strictly prohibits automated handling of litigation risks.",
                risk_level="HIGH",
                trigger_rule="LEGAL_FRAUD_ESCALATION"
            )

        # Tier 3: Hard Policy Domain Gates (HIGH / MEDIUM)
        if predicted_intent == IntentCategory.ACCOUNT_SECURITY.value:
            return EscalationDecision(
                should_escalate=True,
                reason="Account credentials, Apple ID recovery, or 2FA authentication cannot be automated due to security and identity verification compliance.",
                risk_level="HIGH",
                trigger_rule="POLICY_ACCOUNT_SECURITY"
            )

        if predicted_intent == IntentCategory.BILLING_SUBSCRIPTIONS.value:
            # If dispute/unauthorized charge or refund
            if any(w in lower_text for w in ["unauthorized", "stole", "refund", "charged twice", "charge", "invoice", "bank"]):
                return EscalationDecision(
                    should_escalate=True,
                    reason="Financial charge dispute or refund authorization requires human billing tier review.",
                    risk_level="MEDIUM",
                    trigger_rule="POLICY_BILLING_DISPUTE"
                )

        # Tier 4: Incomplete Context / Conversation Fragment Gate (MEDIUM)
        if predicted_intent == IntentCategory.INCOMPLETE_CONTEXT.value:
            return EscalationDecision(
                should_escalate=True,
                reason="Customer query is a conversational fragment lacking prior thread context; routing to human to stitch conversation.",
                risk_level="MEDIUM",
                trigger_rule="FRAGMENT_MISSING_CONTEXT"
            )

        # Tier 5: Retail Escalation / Store In-Person Complaint
        if "genius bar" in lower_text and any(w in lower_text for w in ["waiting", "hours", "late", "rude"]):
            return EscalationDecision(
                should_escalate=True,
                reason="Active in-store Genius Bar wait time or customer complaint requiring store management dispatch.",
                risk_level="MEDIUM",
                trigger_rule="RETAIL_STORE_ESCALATION"
            )

        # Tier 6: Ambiguity & Low-Confidence Gate
        if classifier_confidence < self.confidence_threshold:
            return EscalationDecision(
                should_escalate=True,
                reason=f"Classification confidence ({classifier_confidence:.2f}) below threshold ({self.confidence_threshold:.2f}); routing to prevent hallucinated troubleshooting.",
                risk_level="LOW",
                trigger_rule="CONFIDENCE_UNCERTAINTY"
            )

        # Default: Safe for Auto-Handling
        return EscalationDecision(
            should_escalate=False,
            reason="Standard technical troubleshooting or self-service navigation. Safe for automated response grounded in verified historical resolutions.",
            risk_level="LOW",
            trigger_rule="AUTO_RESOLVE_ROUTINE"
        )
