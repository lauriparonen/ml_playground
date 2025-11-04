import re
from pathlib import Path

def load_vault_texts(vault_path):
    notes = []
    for md in Path(vault_path).rglob("*.md"):
        text = md.read_text(encoding="utf-8", errors="ignore")
        notes.append((md.stem, text))
    return notes

# stopwords from https://gist.github.com/sebleier/554280
STOPWORDS = set(open("stopwords.txt").read().split())

# barebones tokenizer
def tokenize(text):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return [w for w in words if w not in STOPWORDS]
"""
# mathjax-aware tokenizer
def tokenize(text):
    # strip latex/mathjax + code fences
    text = re.sub(r"\$.*?\$", " ", text)        # inline math
    text = re.sub(r"\\\[.*?\\\]", " ", text)    # display math
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # code blocks
    text = re.sub(r"\\[a-zA-Z]+", " ", text)    # latex commands like \frac, \cdot
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return [w for w in words if w not in STOPWORDS]
"""