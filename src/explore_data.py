"""
Explore the Twitter Support dataset to pick the best brand.
Run this first: python src/explore_data.py
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data_loader import load_all_brands_sample, get_brand_summary


def main():
    print("=" * 60)
    print("STEP 1: Loading sample of all brands...")
    print("=" * 60)
    df = load_all_brands_sample(n_rows=200000, split="training")

    print("\n" + "=" * 60)
    print("STEP 2: Brand summary (top 15 brands)")
    print("=" * 60)
    stats = get_brand_summary(df)
    print(stats.head(15).to_string())

    print("\n" + "=" * 60)
    print("STEP 3: Sample tweets from top brands")
    print("=" * 60)
    top_brands = stats.head(5).index.tolist()
    for brand in top_brands:
        brand_df = df[(df["author_id"] == brand) & (df["inbound"] == True)]
        print(f"\n--- {brand} ({len(brand_df)} messages) ---")
        samples = brand_df.sample(min(3, len(brand_df)), random_state=42)
        for _, row in samples.iterrows():
            text = row["text"][:200]
            print(f"  > {text}")

    print("\n" + "=" * 60)
    print("RECOMMENDATION: Pick a brand with:")
    print("  - High message count (>500)")
    print("  - Clear, repeatable patterns")
    print("  - Manageable complexity")
    print("=" * 60)


if __name__ == "__main__":
    main()
