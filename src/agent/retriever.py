"""
retriever.py
Retrieves historically resolved AppleSupport interactions from data/apple_support_corpus.csv
to ground the drafting engine in verified brand resolutions.
"""

import os
from typing import Dict, List, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResolutionRetriever:
    def __init__(self, corpus_path: str = os.path.join("data", "apple_support_corpus.csv")):
        self.corpus_path = corpus_path
        self.df = None
        self.vectorizer = None
        self.corpus_matrix = None
        self._is_indexed = False

    def load_and_index(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus file not found at {self.corpus_path}")

        print(f"Loading historical resolution corpus from {self.corpus_path}...")
        self.df = pd.read_csv(self.corpus_path)
        self.df = self.df.dropna(subset=['customer_text', 'agent_text']).reset_index(drop=True)

        print(f"Indexing {len(self.df)} historical interactions...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=25000,
            sublinear_tf=True
        )
        self.corpus_matrix = self.vectorizer.fit_transform(self.df['customer_text'])
        self._is_indexed = True
        print("Historical corpus indexing complete.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieves top_k closest historical interactions and their verified resolutions.
        """
        if not self._is_indexed:
            self.load_and_index()

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.corpus_matrix).flatten()

        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            results.append({
                "similarity_score": round(score, 4),
                "historical_query": self.df.iloc[idx]['customer_text'],
                "historical_resolution": self.df.iloc[idx]['agent_text']
            })

        return results
