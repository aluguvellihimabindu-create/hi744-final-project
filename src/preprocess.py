import re

def preprocess_text(text: str) -> str:
    """
    Simple baseline preprocessing:
    - lowercase
    - remove non-alphanumeric chars (keep spaces)
    - collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
