"""
Data loader for Hiver SDE Intern Assignment.

Loads Customer Support on Twitter dataset from local CSV (Kaggle download).
Path: D:\dataset\twcs\twcs.csv
"""

import pandas as pd
import os
from typing import Optional

LOCAL_CSV = r"D:\dataset\twcs\twcs.csv"


def load_twitter_support(
    brand: Optional[str] = None,
    split: str = "train",
    streaming: bool = False,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load Customer Support on Twitter from local CSV.

    Args:
        brand: Filter to a specific brand name (e.g. "Amazon", "Apple").
               If None, returns all brands.
        max_rows: Maximum number of rows to load. None = all.
    """
    print(f"Loading Twitter Support dataset from {LOCAL_CSV}...")
    df = pd.read_csv(LOCAL_CSV, nrows=max_rows)
    print(f"  Total rows loaded: {len(df)}")

    if brand:
        df = df[df["author_id"] == brand].copy()
        print(f"  Filtered to brand '{brand}': {len(df)} rows")

    return df


def load_all_brands_sample(
    n_rows: int = 200000, split: str = "train"
) -> pd.DataFrame:
    """
    Load a sample of all brands to explore which brand to use.
    """
    print(f"Loading {n_rows} rows from all brands to explore...")
    df = pd.read_csv(LOCAL_CSV, nrows=n_rows)
    print(f"  Loaded {len(df)} rows across all brands")
    return df


def get_brand_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize brands by message count, reply count, and activity.
    Helps decide which brand to pick.
    """
    # Brand = author_id for incoming messages
    incoming = df[df["inbound"] == True]
    outgoing = df[df["inbound"] == False]

    brand_stats = (
        incoming.groupby("author_id")
        .agg(
            incoming_count=("tweet_id", "count"),
            unique_threads=("in_response_to_tweet_id", "nunique"),
        )
        .sort_values("incoming_count", ascending=False)
        .head(30)
    )

    return brand_stats


def load_banking77(streaming: bool = True) -> pd.DataFrame:
    """
    Load Banking77 from HuggingFace for intent work.
    Contains 77 labelled intent classes.
    """
    print("Loading Banking77 dataset...")
    ds = load_dataset("PolyAI/banking77", streaming=streaming)

    if streaming:
        rows = []
        for row in ds["test"]:
            rows.append(row)
        df = pd.DataFrame(rows)
    else:
        df = ds["test"].to_pandas()

    print(f"  Loaded {len(df)} rows, {df['label'].nunique()} intent classes")
    return df


def build_conversation_threads(df: pd.DataFrame) -> list[dict]:
    """
    Reconstruct multi-turn conversation threads from individual tweets.
    Returns list of dicts with thread_id, messages (list of text + role).
    """
    df = df.sort_values("created_at")

    # Build lookup: tweet_id -> row
    tweet_lookup = {}
    for _, row in df.iterrows():
        tweet_lookup[row["tweet_id"]] = row

    # Find root tweets (not in response to anything)
    root_tweets = df[
        df["in_response_to_tweet_id"].isna()
        | (~df["in_response_to_tweet_id"].isin(df["tweet_id"]))
    ]

    threads = []
    for _, root in root_tweets.iterrows():
        thread = {
            "thread_id": root["tweet_id"],
            "messages": [],
        }

        # Add root message (customer)
        thread["messages"].append(
            {"text": root["text"], "role": "customer", "tweet_id": root["tweet_id"]}
        )

        # Find all replies in chain
        current_id = root["tweet_id"]
        visited = {current_id}

        while True:
            replies = df[df["in_response_to_tweet_id"] == current_id]
            if replies.empty:
                break

            reply = replies.iloc[0]  # Take first reply
            if reply["tweet_id"] in visited:
                break
            visited.add(reply["tweet_id"])

            role = "agent" if not reply["inbound"] else "customer"
            thread["messages"].append(
                {"text": reply["text"], "role": role, "tweet_id": reply["tweet_id"]}
            )
            current_id = reply["tweet_id"]

        threads.append(thread)

    return threads


def save_brand_data(df: pd.DataFrame, brand: str, output_dir: str = "data"):
    """Save filtered brand data to CSV."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{brand.lower().replace(' ', '_')}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved to {path}")
    return path
