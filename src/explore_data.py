"""
Explore the Twitter Support dataset to pick the best brand.
Run this first: python src/explore_data.py
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import ui
from src.data_loader import load_all_brands_sample, get_brand_summary


def main():
    ui.banner("STEP 1: Loading sample of all brands")
    df = load_all_brands_sample(n_rows=200000, split="training")

    ui.subheader("STEP 2: Brand summary (top 15 brands)")
    stats = get_brand_summary(df)
    rows = [
        (str(b), s.incoming_count, s.unique_threads)
        for b, s in stats.head(15).iterrows()
    ]
    ui.table(["Brand", "Incoming msgs", "Unique threads"], rows)

    ui.subheader("STEP 3: Sample tweets from top brands")
    top_brands = stats.head(5).index.tolist()
    for brand in top_brands:
        brand_df = df[(df["author_id"] == brand) & (df["inbound"] == True)]
        print(f"\n  --- {brand} ({len(brand_df)} messages) ---")
        samples = brand_df.sample(min(3, len(brand_df)), random_state=42)
        for _, row in samples.iterrows():
            print(f"    > {row['text'][:200]}")

    ui.subheader("RECOMMENDATION: pick a brand with")
    ui.kv_pairs(
        {
            "High message count": ">500",
            "Clear patterns": "repeatable, common issues",
            "Manageable complexity": "few, well-defined intents",
        }
    )


if __name__ == "__main__":
    main()