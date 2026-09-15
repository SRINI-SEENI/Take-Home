"""
taxonomy.py
Defines the intent hierarchy, descriptions, risk profiles, and escalation criteria
for the @AppleSupport customer support domain.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class IntentCategory(str, Enum):
    BATTERY_POWER = "battery_power_hardware"
    SOFTWARE_OS = "software_os_update"
    ACCOUNT_SECURITY = "account_security_appleid"
    BILLING_SUBSCRIPTIONS = "billing_subscriptions_refund"
    DEVICE_BACKUP = "device_backup_sync"
    REPAIR_HARDWARE = "repair_applecare_hardware"
    CHITCHAT_RANT = "chitchat_feedback_rant"
    INCOMPLETE_CONTEXT = "incomplete_context"
    OUT_OF_SCOPE = "out_of_scope"

@dataclass
class IntentMetadata:
    name: str
    description: str
    mandatory_escalate: bool
    typical_action: str
    sample_keywords: List[str]

TAXONOMY: Dict[str, IntentMetadata] = {
    IntentCategory.BATTERY_POWER: IntentMetadata(
        name="Battery & Power Hardware",
        description="Battery drain, unexpected shutdowns, charging cable/port issues, device overheating.",
        mandatory_escalate=False,
        typical_action="Auto-troubleshoot: Guide user to Settings > Battery > Battery Health and clean charging port.",
        sample_keywords=["battery", "drain", "charge", "charger", "charging", "overheat", "shut down", "dies", "percentage", "mah"]
    ),
    IntentCategory.SOFTWARE_OS: IntentMetadata(
        name="Software & OS Updates",
        description="Bugs after iOS/macOS update, keyboard lag, app crashes, Wi-Fi or Bluetooth connectivity issues.",
        mandatory_escalate=False,
        typical_action="Auto-troubleshoot: Reset network settings, force restart device, update to latest patch.",
        sample_keywords=["update", "ios", "macos", "lag", "freezing", "wifi", "bluetooth", "safari", "keyboard", "glitch", "crash"]
    ),
    IntentCategory.ACCOUNT_SECURITY: IntentMetadata(
        name="Account Security & Apple ID",
        description="Locked Apple ID, two-factor authentication failure, lost passcodes, activation lock, device theft.",
        mandatory_escalate=True,
        typical_action="Mandatory Escalation: Secure DM routing or account recovery referral at iforgot.apple.com.",
        sample_keywords=["apple id", "password", "locked", "passcode", "disabled", "two-factor", "2fa", "verification code", "stolen", "activation lock"]
    ),
    IntentCategory.BILLING_SUBSCRIPTIONS: IntentMetadata(
        name="Billing & Subscriptions",
        description="App Store unrecognized charges, refund requests, subscription cancellations, invoice disputes.",
        mandatory_escalate=True,
        typical_action="Mandatory Escalation: Route to verified billing support; bot cannot inspect credit card transactions.",
        sample_keywords=["refund", "subscription", "charge", "charged", "invoice", "receipt", "in-app", "credit card", "billed", "cancel subscription"]
    ),
    IntentCategory.DEVICE_BACKUP: IntentMetadata(
        name="Device Backup & Syncing",
        description="iCloud storage full, backup failures, migrating data between devices, photo/contact sync.",
        mandatory_escalate=False,
        typical_action="Auto-troubleshoot: Quick Start guide, check iCloud storage breakdown, toggle iCloud Drive.",
        sample_keywords=["backup", "icloud", "sync", "transfer", "itunes", "contacts", "photos", "quick start", "storage full"]
    ),
    IntentCategory.REPAIR_HARDWARE: IntentMetadata(
        name="Repair, AppleCare & Hardware",
        description="Cracked screens, water damage, speaker/mic failure, Genius Bar appointments, AppleCare coverage.",
        mandatory_escalate=False,
        typical_action="Auto-troubleshoot: Link to trade-in/AppleCare lookup or schedule Genius Bar appointment.",
        sample_keywords=["screen", "cracked", "repair", "genius bar", "appointment", "applecare", "broken", "speaker", "water damage", "microphone"]
    ),
    IntentCategory.CHITCHAT_RANT: IntentMetadata(
        name="Feedback & Brand Rants",
        description="Customer venting, sarcastic praise, pricing complaints without explicit request for diagnostic assistance.",
        mandatory_escalate=False,
        typical_action="De-escalate: Empathetic acknowledgement, invite user to DM if they want targeted assistance.",
        sample_keywords=["worst", "hate", "terrible", "sucks", "tim cook", "useless", "expensive", "rip off", "switching"]
    ),
    IntentCategory.INCOMPLETE_CONTEXT: IntentMetadata(
        name="Incomplete Context / Fragment",
        description="Follow-up responses ('yes', '11.2', 'DM sent') with no antecedent in the current tweet.",
        mandatory_escalate=True,
        typical_action="Mandatory Escalation: Needs agent thread stitching or prompt for full context.",
        sample_keywords=["yes", "no", "dm sent", "still doing it", "same here", "done"]
    ),
    IntentCategory.OUT_OF_SCOPE: IntentMetadata(
        name="Out of Scope / Spam",
        description="Inquiries regarding Android, Windows, unrelated commercial brands, spam links.",
        mandatory_escalate=False,
        typical_action="Decline politely: Clarify Apple Support domain scope.",
        sample_keywords=["samsung", "android", "windows", "pizza", "uber", "soundcloud", "mixtape"]
    )
}

HIGH_STAKES_KEYWORDS = [
    "lawyer", "attorney", "sue", "legal", "police", "fraud", "stole", "steal",
    "burned", "fire", "smoke", "exploded", "melted", "hazard", "court"
]
