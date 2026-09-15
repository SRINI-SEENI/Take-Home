"""
reply_drafter.py
Generates customer support replies grounded in historically resolved @AppleSupport cases.
Enforces official Apple brand guidelines:
- Empathetic, professional, and courteous opening
- Action-oriented diagnostic steps or official Apple Support resource links
- Strict adherence to Twitter character limitations (<= 280 chars)
- Safe routing for escalated cases (directing to secure DM)
"""

import re
from typing import Dict, List, Optional
from src.agent.escalator import EscalationDecision
from src.agent.taxonomy import IntentCategory

class ReplyDrafter:
    def __init__(self):
        pass

    def clean_brand_text(self, text: str) -> str:
        """Removes external Twitter customer handle numbers from historical tweets."""
        cleaned = re.sub(r'@\d+', '', text).strip()
        cleaned = re.sub(r'https?://t\.co/\w+', 'https://apple.co/support', cleaned)
        cleaned = cleaned.replace('', "'").replace('’', "'")
        return re.sub(r'\s+', ' ', cleaned).strip()

    def draft(
        self,
        customer_text: str,
        predicted_intent: str,
        escalation: EscalationDecision,
        historical_resolutions: List[Dict]
    ) -> str:
        # Case A: Escalated cases require safe handoff to DM / Human Specialist
        if escalation.should_escalate:
            if escalation.trigger_rule == "SAFETY_PHYSICAL_HAZARD":
                return (
                    "Your safety is our absolute priority. Please disconnect and stop using the device "
                    "immediately. Send us a DM with your contact details so our safety engineering team "
                    "can assist you directly."
                )[:280]

            if escalation.trigger_rule == "LEGAL_FRAUD_ESCALATION":
                return (
                    "We take these concerns very seriously and want to review your case directly. "
                    "Please send us a DM with your case number and account email so a senior specialist can assist."
                )[:280]

            if escalation.trigger_rule == "POLICY_ACCOUNT_SECURITY":
                return (
                    "We understand the importance of securing your Apple ID. To protect your personal "
                    "credentials, please reach out to us via DM or visit https://iforgot.apple.com to begin recovery."
                )[:280]

            if escalation.trigger_rule == "POLICY_BILLING_DISPUTE":
                return (
                    "We'd like to help investigate this billing charge. For account security, please send us a DM "
                    "with your Apple ID email so our billing team can review your purchase history."
                )[:280]

            if escalation.trigger_rule == "FRAGMENT_MISSING_CONTEXT":
                return (
                    "Thanks for reaching back out! Could you clarify which issue you are experiencing, "
                    "along with your device model and iOS version? We're here to help."
                )[:280]

            # General escalation
            return (
                "We want to make sure you get the best possible help with this. Please send us a direct message "
                "so we can look into your device details and assist you further."
            )[:280]

        # Case B: Auto-handle with Grounded Historical Retrieval
        best_resolution = ""
        if historical_resolutions:
            best_resolution = self.clean_brand_text(historical_resolutions[0].get("historical_resolution", ""))

        # If retrieved resolution is actionable, adapt it
        if best_resolution and len(best_resolution) > 25:
            # Check if it has an actionable instruction or link
            if any(w in best_resolution.lower() for w in ["settings", "restart", "press", "guide", "steps", "visit"]):
                # Ensure it fits Twitter length
                if len(best_resolution) <= 275:
                    return best_resolution

        # Domain-specific verified fallback resolutions grounded in Apple documentation
        intent_grounded_templates = {
            IntentCategory.BATTERY_POWER.value: (
                "We know how crucial battery life is. Head to Settings > Battery > Battery Health "
                "to verify maximum capacity. If below 80%, a battery service may be recommended: https://apple.co/support"
            ),
            IntentCategory.SOFTWARE_OS.value: (
                "Let's work to get this running smoothly. Try a force restart on your device, and test "
                "Settings > General > Reset > Reset All Settings if the glitch continues: https://apple.co/support"
            ),
            IntentCategory.DEVICE_BACKUP.value: (
                "We can guide you through managing your backups. Check Settings > [Your Name] > iCloud > "
                "iCloud Backup to verify your latest backup status: https://support.apple.com/HT203977"
            ),
            IntentCategory.REPAIR_HARDWARE.value: (
                "We want to help get your device repaired. You can view official repair pricing and book "
                "a Genius Bar appointment directly through: https://support.apple.com/repair"
            ),
            IntentCategory.CHITCHAT_RANT.value: (
                "We hear you, and we appreciate your feedback. If you're experiencing a specific technical issue "
                "with your device, let us know your model and iOS version—we'd love to help."
            ),
            IntentCategory.OUT_OF_SCOPE.value: (
                "We specialize in supporting Apple hardware and software services. For third-party or non-Apple "
                "products, please consult the respective manufacturer's support team."
            )
        }

        reply = intent_grounded_templates.get(
            predicted_intent,
            "We're here to help! Please let us know which device model and iOS version you're using so we can assist."
        )
        return reply[:280]
