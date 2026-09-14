"""
extract_apple_support.py
Extracts and reconstructs single-turn inquiry-response pairs for @AppleSupport
from the Kaggle Customer Support on Twitter dataset (twcs.csv).

Produces a clean, compact dataset (data/apple_support_corpus.csv) suitable
for quick local development, retrieval-augmented generation, and evaluation.
"""

import os
import re
import pandas as pd
from typing import Dict, List, Optional

RAW_PATH = os.path.join("data", "twcs.csv")
OUTPUT_CORPUS = os.path.join("data", "apple_support_corpus.csv")
MAX_ROWS_SCAN = 800_000  # Scan enough chunks to collect rich historical interactions
MAX_PAIRS = 15_000       # Targeted corpus size for fast, sub-minute reproduction

def clean_tweet_text(text: str) -> str:
    """Normalizes tweet text: strips extraneous whitespace and standardizes handle tokens."""
    if not isinstance(text, str):
        return ""
    # Normalize unicode spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_pairs():
    print(f"Reading raw dataset from {RAW_PATH}...")
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Missing {RAW_PATH}. Please ensure twcs.csv is present.")

    # Read needed columns in chunks to be memory efficient
    chunks = pd.read_csv(
        RAW_PATH,
        usecols=['tweet_id', 'author_id', 'inbound', 'text', 'response_tweet_id', 'in_response_to_tweet_id'],
        nrows=MAX_ROWS_SCAN,
        chunksize=100_000,
        low_memory=False
    )

    df_list = []
    for i, chunk in enumerate(chunks):
        # Filter for AppleSupport tweets and inbound tweets
        filtered = chunk[
            (chunk['author_id'] == 'AppleSupport') |
            (chunk['text'].str.contains('@AppleSupport', case=False, na=False))
        ]
        df_list.append(filtered)
        print(f"Processed chunk {i+1}, found {len(filtered)} relevant tweets.")

    df = pd.concat(df_list, ignore_index=True)
    print(f"Total relevant tweets collected: {len(df)}")

    # Index by tweet_id for fast lookup
    tweet_dict = df.set_index('tweet_id').to_dict(orient='index')

    pairs = []
    # Identify AppleSupport agent responses and link them to the preceding customer tweet
    for tweet_id, row in tweet_dict.items():
        if row['author_id'] == 'AppleSupport' and not row['inbound']:
            parent_id = row['in_response_to_tweet_id']
            if pd.notna(parent_id) and parent_id in tweet_dict:
                parent_tweet = tweet_dict[parent_id]
                # Verify the parent was inbound from a customer
                if parent_tweet['inbound']:
                    cust_text = clean_tweet_text(parent_tweet['text'])
                    agent_text = clean_tweet_text(row['text'])
                    
                    # Basic noise filters
                    if len(cust_text) >= 15 and len(agent_text) >= 15:
                        pairs.append({
                            'inquiry_id': parent_id,
                            'response_id': tweet_id,
                            'customer_text': cust_text,
                            'agent_text': agent_text
                        })

        if len(pairs) >= MAX_PAIRS:
            break

    pair_df = pd.DataFrame(pairs).drop_duplicates(subset=['customer_text'])
    os.makedirs("data", exist_ok=True)
    pair_df.to_csv(OUTPUT_CORPUS, index=False, encoding='utf-8')
    print(f"Successfully extracted {len(pair_df)} clean pairs to {OUTPUT_CORPUS}")

if __name__ == "__main__":
    extract_pairs()
