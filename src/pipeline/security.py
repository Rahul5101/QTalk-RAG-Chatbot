"""
Security Module for RAG Pipeline.
Includes Input Sanitization, Prompt Injection Detection, and PII Masking.
"""

import re
import html
from typing import Tuple, Dict, Any, List

# Regex Patterns for PII
EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
PHONE_REGEX = r'(\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}'
SSN_REGEX = r'\b\d{3}-\d{2}-\d{4}\b'
CREDIT_CARD_REGEX = r'\b(?:\d[ -]*?){13,16}\b'
API_KEY_REGEX = r'\b(?:sk-[a-zA-Z0-9]{32,}|AIzaSy[a-zA-Z0-9_-]{33}|AQ\.[a-zA-Z0-9_-]{30,})\b'
IPV4_REGEX = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'

# Heuristic Patterns for Prompt Injection / Jailbreaks
INJECTION_PATTERNS = [
    r'ignore\s+all\s+previous\s+instructions',
    r'ignore\s+above\s+instructions',
    r'disregard\s+prior\s+prompts',
    r'you\s+are\s+now\s+a\s+DAN',
    r'do\s+anything\s+now',
    r'system\s*:\s*',
    r'<\s*system\s*>',
    r'override\s+system\s+prompt',
    r'jailbreak',
    r'reveal\s+your\s+system\s+instructions',
    r'output\s+the\s+secret\s+key',
    r'bypass\s+restrictions'
]

class SecurityGuard:
    def __init__(self):
        self.injection_regexes = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

    def sanitize_input(self, text: str) -> str:
        """
        Sanitizes user input by escaping HTML and stripping unprintable control characters.
        """
        if not text:
            return ""
        # Remove null bytes and unprintable ASCII control characters (except newline, tab, carriage return)
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        # Escape potential script tags / HTML
        cleaned = html.escape(cleaned.strip())
        return cleaned

    def check_prompt_injection(self, text: str) -> Tuple[bool, str]:
        """
        Checks if the input text contains prompt injection or jailbreak patterns.
        Returns (is_injected, matched_pattern).
        """
        if not text:
            return False, ""
        
        for regex in self.injection_regexes:
            match = regex.search(text)
            if match:
                return True, match.group(0)
        return False, ""

    def mask_pii(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Masks PII elements in input/output text.
        Returns (masked_text, counts_dict).
        """
        if not text:
            return "", {}

        masked = text
        counts = {
            "email": 0,
            "phone": 0,
            "ssn": 0,
            "credit_card": 0,
            "api_key": 0,
            "ip_address": 0
        }

        # Mask API Keys first to avoid false overlaps
        masked, count_key = re.subn(API_KEY_REGEX, '[REDACTED_API_KEY]', masked)
        counts["api_key"] = count_key

        masked, count_email = re.subn(EMAIL_REGEX, '[REDACTED_EMAIL]', masked)
        counts["email"] = count_email

        masked, count_ssn = re.subn(SSN_REGEX, '[REDACTED_SSN]', masked)
        counts["ssn"] = count_ssn

        masked, count_cc = re.subn(CREDIT_CARD_REGEX, '[REDACTED_CARD]', masked)
        counts["credit_card"] = count_cc

        masked, count_phone = re.subn(PHONE_REGEX, '[REDACTED_PHONE]', masked)
        counts["phone"] = count_phone

        masked, count_ip = re.subn(IPV4_REGEX, '[REDACTED_IP]', masked)
        counts["ip_address"] = count_ip

        return masked, counts

security_guard = SecurityGuard()
