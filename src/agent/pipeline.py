"""
pipeline.py
Unified agent pipeline orchestrating Intent Classification, Historical Resolution
Retrieval, Multi-Tier Escalation, and Reply Drafting.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from src.agent.classifier import (
    TrivialBaselineClassifier,
    SimpleBaselineClassifier,
    ProductionClassifier
)
from src.agent.retriever import ResolutionRetriever
from src.agent.escalator import EscalationEngine, EscalationDecision
from src.agent.reply_drafter import ReplyDrafter

@dataclass
class AgentOutput:
    customer_text: str
    intent: str
    confidence: float
    should_escalate: bool
    escalation_reason: str
    risk_level: str
    drafted_reply: str
    retrieved_resolutions: List[Dict]

class SupportAgentPipeline:
    def __init__(self, mode: str = "production"):
        """
        mode: 'production', 'simple_baseline', or 'trivial_baseline'
        """
        self.mode = mode
        self.retriever = ResolutionRetriever()
        self.escalator = EscalationEngine()
        self.drafter = ReplyDrafter()

        if mode == "trivial_baseline":
            self.classifier = TrivialBaselineClassifier()
        elif mode == "simple_baseline":
            self.classifier = SimpleBaselineClassifier()
        else:
            self.classifier = ProductionClassifier()

    def train_classifier(self, training_texts: List[str], training_labels: List[str]):
        """Fits the classifier on training data."""
        self.classifier.fit(training_texts, training_labels)

    def process_message(self, customer_text: str) -> AgentOutput:
        # Step 1: Intent Classification
        if self.mode == "trivial_baseline":
            intent, confidence = self.classifier.predict_one(customer_text)
            # Trivial escalation: never escalate
            escalation = EscalationDecision(
                should_escalate=False,
                reason="Trivial baseline: default auto-handle all queries.",
                risk_level="LOW",
                trigger_rule="TRIVIAL_DEFAULT"
            )
            drafted_reply = "Thanks for reaching out! Please DM us your Apple ID and iOS version."
            retrieved = []
            return AgentOutput(
                customer_text=customer_text,
                intent=intent,
                confidence=confidence,
                should_escalate=escalation.should_escalate,
                escalation_reason=escalation.reason,
                risk_level=escalation.risk_level,
                drafted_reply=drafted_reply,
                retrieved_resolutions=retrieved
            )

        elif self.mode == "simple_baseline":
            intent, confidence = self.classifier.predict_one(customer_text)
            # Simple keyword escalation rule
            lower = customer_text.lower()
            if any(w in lower for w in ["refund", "money", "stolen", "lawyer", "hack"]):
                should_esc = True
                reason = "Simple keyword escalation match."
            else:
                should_esc = False
                reason = "Simple baseline: no trigger keywords detected."

            escalation = EscalationDecision(
                should_escalate=should_esc,
                reason=reason,
                risk_level="MEDIUM" if should_esc else "LOW",
                trigger_rule="KEYWORD_RULE"
            )
            # 1-NN verbatim retrieval
            retrieved = self.retriever.retrieve(customer_text, top_k=1)
            raw_reply = retrieved[0]["historical_resolution"] if retrieved else "We are here to help. Send us a DM."
            return AgentOutput(
                customer_text=customer_text,
                intent=intent,
                confidence=confidence,
                should_escalate=escalation.should_escalate,
                escalation_reason=escalation.reason,
                risk_level=escalation.risk_level,
                drafted_reply=raw_reply[:280],
                retrieved_resolutions=retrieved
            )

        else: # Production mode
            intent, confidence = self.classifier.predict_one(customer_text)
            retrieved = self.retriever.retrieve(customer_text, top_k=3)
            retrieval_score = retrieved[0]["similarity_score"] if retrieved else 0.0

            escalation = self.escalator.decide(
                customer_text=customer_text,
                predicted_intent=intent,
                classifier_confidence=confidence,
                retrieval_score=retrieval_score
            )

            drafted_reply = self.drafter.draft(
                customer_text=customer_text,
                predicted_intent=intent,
                escalation=escalation,
                historical_resolutions=retrieved
            )

            return AgentOutput(
                customer_text=customer_text,
                intent=intent,
                confidence=confidence,
                should_escalate=escalation.should_escalate,
                escalation_reason=escalation.reason,
                risk_level=escalation.risk_level,
                drafted_reply=drafted_reply,
                retrieved_resolutions=retrieved
            )
