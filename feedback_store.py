"""
feedback_store.py
Stage 4: Feedback-Aware Re-ranking.

Stores user feedback (correct/incorrect) per query result.
Uses TF-IDF cosine similarity to find similar past queries
and biases confidence scores accordingly.

No external vector DB needed — pure Python + sklearn TF-IDF.
"""

import json
import math
import re
from pathlib import Path
from datetime import datetime
from typing import Optional


FEEDBACK_FILE = Path(__file__).parent / "data" / "feedback_log.json"


def _tokenize(text: str) -> list:
    """Simple tokenizer — lowercase words, no stopwords."""
    return re.findall(r"\b[a-z]+\b", text.lower())


def _tfidf_similarity(query1: str, query2: str, corpus: list) -> float:
    """Compute TF-IDF cosine similarity between two queries given a corpus."""
    all_docs = corpus + [query1, query2]
    
    # Build vocabulary
    vocab = {}
    for doc in all_docs:
        for word in _tokenize(doc):
            if word not in vocab:
                vocab[word] = len(vocab)
    
    def tf(doc_tokens: list) -> dict:
        freq = {}
        for w in doc_tokens:
            freq[w] = freq.get(w, 0) + 1
        return {w: c / len(doc_tokens) for w, c in freq.items()} if doc_tokens else {}
    
    def idf(word: str) -> float:
        n_docs_with_word = sum(1 for doc in all_docs if word in _tokenize(doc))
        return math.log((len(all_docs) + 1) / (n_docs_with_word + 1)) + 1
    
    def tfidf_vec(text: str) -> dict:
        tokens = _tokenize(text)
        tf_map = tf(tokens)
        return {w: tf_map.get(w, 0) * idf(w) for w in vocab}
    
    v1 = tfidf_vec(query1)
    v2 = tfidf_vec(query2)
    
    dot = sum(v1[w] * v2[w] for w in vocab)
    mag1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    mag2 = math.sqrt(sum(x ** 2 for x in v2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


class FeedbackStore:
    def __init__(self):
        self.entries = []
        self._load()
    
    def _load(self):
        """Load existing feedback from file."""
        if FEEDBACK_FILE.exists():
            try:
                with open(FEEDBACK_FILE) as f:
                    self.entries = json.load(f)
            except Exception:
                self.entries = []
    
    def _save(self):
        """Persist feedback to file."""
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(self.entries, f, indent=2)
    
    def add_feedback(
        self,
        query: str,
        is_correct: bool,
        intent_type: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        """Record user feedback for a query result."""
        entry = {
            "query": query,
            "is_correct": is_correct,
            "intent_type": intent_type,
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.entries.append(entry)
        self._save()
        return entry
    
    def get_feedback_bias(self, query: str, threshold: float = 0.4) -> dict:
        """
        Find similar past queries and compute a feedback bias.
        Returns a bias dict with:
        - score_adjustment: float to add/subtract from confidence
        - similar_queries: list of similar past queries with their feedback
        """
        if not self.entries:
            return {"score_adjustment": 0.0, "similar_queries": []}
        
        corpus = [e["query"] for e in self.entries]
        
        similar = []
        for entry in self.entries:
            sim = _tfidf_similarity(query, entry["query"], corpus)
            if sim >= threshold:
                similar.append({
                    "query": entry["query"],
                    "similarity": round(sim, 3),
                    "was_correct": entry["is_correct"],
                    "timestamp": entry.get("timestamp", ""),
                })
        
        if not similar:
            return {"score_adjustment": 0.0, "similar_queries": []}
        
        # Weight by similarity
        total_weight = sum(s["similarity"] for s in similar)
        weighted_correct = sum(
            s["similarity"] * (1 if s["was_correct"] else -1)
            for s in similar
        )
        
        # Adjustment range: -0.2 to +0.2
        raw_adj = (weighted_correct / total_weight) * 0.2
        adjustment = max(-0.2, min(0.2, raw_adj))
        
        return {
            "score_adjustment": round(adjustment, 3),
            "similar_queries": sorted(similar, key=lambda x: -x["similarity"])[:3],
        }
    
    def get_all_feedback(self) -> list:
        return self.entries
    
    def get_stats(self) -> dict:
        if not self.entries:
            return {"total": 0, "correct": 0, "incorrect": 0, "accuracy": None}
        correct = sum(1 for e in self.entries if e["is_correct"])
        return {
            "total": len(self.entries),
            "correct": correct,
            "incorrect": len(self.entries) - correct,
            "accuracy": round(correct / len(self.entries), 3),
        }
