"""
reply_drafter.py
Generates customer support replies grounded in historically resolved @AppleSupport cases.
Supports:
1. Real-time LLM generation (via Gemini, OpenAI, or Groq API if keys are provided)
2. Dynamic contextual resolution synthesis (extracting specific symptoms: cable/port,
   battery wear, thermal heat, Wi-Fi drops, storage sync, cracked screens)
Enforces official Apple brand guidelines:
- Empathetic, professional, and courteous opening
- Action-oriented diagnostic steps or official Apple Support resource links
- Strict adherence to Twitter character limitations (<= 280 chars)
- Safe routing for escalated cases (directing to secure DM)
"""

import os
import re
from typing import Dict, List, Optional
import urllib.request
import json
from src.agent.escalator import EscalationDecision
from src.agent.taxonomy import IntentCategory

class ReplyDrafter:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")

    def clean_brand_text(self, text: str) -> str:
        """Removes external Twitter customer handle numbers from historical tweets."""
        cleaned = re.sub(r'@\d+', '', text).strip()
        cleaned = re.sub(r'https?://t\.co/\w+', 'https://apple.co/support', cleaned)
        cleaned = cleaned.replace('', "'").replace('’', "'")
        return re.sub(r'\s+', ' ', cleaned).strip()

    def generate_with_llm(self, customer_text: str, historical_context: str) -> Optional[str]:
        """Calls LLM API if key is available in environment."""
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None

        prompt = (
            f"You are the official @AppleSupport Twitter AI agent.\n"
            f"Customer tweet: {customer_text}\n"
            f"Historical verified resolution: {historical_context}\n"
            f"Draft a response adhering to:\n"
            f"- Polite, empathetic, and professional Apple voice\n"
            f"- Directly solve the specific question (e.g. cable vs port vs battery)\n"
            f"- Include official Apple support link where applicable\n"
            f"- STRICTLY under 280 characters.\n"
            f"Return ONLY the tweet text."
        )

        try:
            # Gemini API integration
            if os.environ.get("GEMINI_API_KEY"):
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={os.environ.get('GEMINI_API_KEY')}"
                payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
                req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return reply[:280]

            # OpenAI API integration
            if os.environ.get("OPENAI_API_KEY"):
                url = "https://api.openai.com/v1/chat/completions"
                payload = json.dumps({
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100
                }).encode("utf-8")
                req = urllib.request.Request(url, data=payload, headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
                })
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    reply = data["choices"][0]["message"]["content"].strip()
                    return reply[:280]
        except Exception:
            return None

        return None

    def draft(
        self,
        customer_text: str,
        predicted_intent: str,
        escalation: EscalationDecision,
        historical_resolutions: List[Dict]
    ) -> str:
        lower_cust = customer_text.lower()

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

        # Case B: Auto-handle with Real-Time Grounding
        best_resolution = ""
        if historical_resolutions:
            best_resolution = self.clean_brand_text(historical_resolutions[0].get("historical_resolution", ""))

        # Check if live LLM generation is available
        llm_reply = self.generate_with_llm(customer_text, best_resolution)
        if llm_reply and len(llm_reply) > 20:
            return llm_reply[:280]

        # Case C: Contextual Entity-Specific Resolution Synthesis
        # Strip handles so @AppleSupport doesn't match 'port'
        clean_cust = re.sub(r'@\w+', ' ', lower_cust)

        # 1. Thermal Overheating / Device Hot (Strict word boundaries so 'photos' never matches 'hot')
        if re.search(r'\b(hot|warm|boiling|overheating|temperature|overheat)\b', clean_cust):
            return (
                "Devices can warm up during heavy gaming or fast charging. If you see a temperature warning, "
                "let it cool in a shaded environment: https://support.apple.com/HT201678"
            )[:280]

        # 2. Physical Charging Cable / Lightning Port / Bending Cable (Explicit hardware cable queries)
        if (re.search(r'\b(cable|cord|bend|bending|lightning|plug|port)\b', clean_cust) or
            re.search(r"\b(won't charge|not charging|stopped charging|refuses to charge)\b", clean_cust)):
            return (
                "We'd like to help you get powered up! Test with an alternate Apple-certified cable first, "
                "and gently inspect your charging port with a flashlight for any pocket lint or debris: https://support.apple.com/HT201569"
            )[:280]

        # 3. Battery Percentage Jumps / Rapid Degradation / Sudden Shutdowns
        if re.search(r'\b(battery|percentage|drops from|drain|draining|dies at|degradation|battery health|capacity)\b', clean_cust):
            return (
                "We know how crucial battery life is. Head to Settings > Battery > Battery Health "
                "to verify maximum capacity. If below 80%, a battery service may be recommended: https://apple.co/support"
            )[:280]

        # 4. Wi-Fi / Bluetooth / Cellular Drops
        if re.search(r'\b(wifi|wi-fi|bluetooth|cellular|no service|lte|airplane mode)\b', clean_cust) or "greyed out" in clean_cust:
            return (
                "Let's work on getting your connection restored. Try going to Settings > General > Reset > "
                "Reset Network Settings. Your device will restart: https://apple.co/support"
            )[:280]

        # 5. Keyboard Lag / App Crashing / Screen Freeze
        if re.search(r'\b(lag|lagging|freeze|freezing|crash|crashing|stuck|unresponsive)\b', clean_cust):
            return (
                "Let's work to get this running smoothly. Try a force restart on your device, and test "
                "Settings > General > Reset > Reset All Settings if the glitch continues: https://apple.co/support"
            )[:280]

        # 6. iCloud / Data Migration / Quick Start / Contacts / Photos
        if re.search(r'\b(transfer|backup|icloud|quick start|contacts|photos|migrate|restore)\b', clean_cust) or predicted_intent == IntentCategory.DEVICE_BACKUP.value:
            return (
                "We can guide you through managing your data. Check Settings > [Your Name] > iCloud > "
                "iCloud Backup, or use Quick Start to migrate directly: https://support.apple.com/HT201269"
            )[:280]

        # 7. Hardware Damage / Cracked Screen / Genius Bar
        if re.search(r'\b(screen|cracked|shattered|repair|genius bar|broken)\b', clean_cust) or predicted_intent == IntentCategory.REPAIR_HARDWARE.value:
            return (
                "We want to help get your device repaired. You can view official repair pricing and book "
                "a Genius Bar appointment directly through: https://support.apple.com/repair"
            )[:280]

        # 8. Subscription Management / Inquiries
        if re.search(r'\b(cancel|subscription|trial|music|renew)\b', clean_cust) or predicted_intent == IntentCategory.BILLING_SUBSCRIPTIONS.value:
            return (
                "We can help you manage your subscriptions. Go to Settings > [Your Name] > Subscriptions "
                "to view or cancel active plans: https://support.apple.com/HT202039"
            )[:280]

        # 9. Brand Feedback / Chitchat
        if predicted_intent == IntentCategory.CHITCHAT_RANT.value:
            return (
                "We hear you, and we appreciate your feedback. If you're experiencing a specific technical issue "
                "with your device, let us know your model and iOS version—we'd love to help."
            )[:280]

        # 10. Out of Scope
        if predicted_intent == IntentCategory.OUT_OF_SCOPE.value:
            return (
                "We specialize in supporting Apple hardware and software services. For third-party or non-Apple "
                "products, please consult the respective manufacturer's support team."
            )[:280]

        # Fallback to high-confidence historical resolution if available
        if best_resolution and len(best_resolution) > 25 and len(best_resolution) <= 275:
            return best_resolution

        return "We're here to help! Please let us know which device model and iOS version you're using so we can assist."[:280]
