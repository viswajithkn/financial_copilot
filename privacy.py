import re
import os

sensitive_words = os.getenv('SENSITIVE_WORDS')
credit_card_pattern = os.getenv('CREDIT_CARD_PATTERN')
account_pattern = os.getenv('ACCOUNT_PATTERN')


class PIIMasker:
    def __init__(self):
        self.mask_vault = {}
        self.counter = 1
        
        # This builds a single regex pattern like: \b(Viswajith|Karapoondi|Madhuram...)\b
        self.names_pattern = re.compile(
            r'\b(' + '|'.join(map(re.escape, sensitive_words)) + r')\b', 
            re.IGNORECASE
        )
        
        # Pre-compile the financial numbers rule (8 to 16 digits)
        self.numbers_pattern = re.compile(r'\b\d{8,16}\b')

        self.card_pattern = re.compile(credit_card_pattern,re.IGNORECASE)
        self.account_pattern = re.compile(account_pattern,re.IGNORECASE)

    def _mask_match(self, match, prefix):
        """Helper to quickly swap text and record it in our vault."""
        val = match.group(0)
        # Check if we already have a key for this exact value
        for k, v in self.mask_vault.items():
            if v == val:
                return k
                
        placeholder = f"[{prefix}_{self.counter}]"
        self.mask_vault[placeholder] = val
        self.counter += 1
        return placeholder

    def mask_text(self, text: str) -> str:
        """Scans the text instantly in only two quick steps."""
        if not text or not isinstance(text, str):
            return text

        # 1. Sweep all numbers at once
        text = self.numbers_pattern.sub(lambda m: self._mask_match(m, "NUM"), text)
        text = self.card_pattern.sub(lambda m: self._mask_match(m, "NUM"), text)
        text = self.account_pattern.sub(lambda m: self._mask_match(m, "NUM"), text)
        
        # 2. Sweep all names at once
        text = self.names_pattern.sub(lambda m: self._mask_match(m, "NAME"), text)
        
        return text

    def unmask_text(self, text: str) -> str:
        """Restores values instantly using a single fast regex swap."""
        if not text or not self.mask_vault:
            return text

        # Build a regex pattern out of all the keys in our vault
        # Example: (\[NUM_1\]|\[NAME_2\])
        pattern = re.compile('|'.join(map(re.escape, self.mask_vault.keys())))
        
        # Swap them back in a single pass
        return pattern.sub(lambda m: self.mask_vault[m.group(0)], text)
        
