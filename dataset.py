"""
Module 1: Data Collection

Loads the CNN/DailyMail (3.0.0) dataset from HuggingFace.
Extracts article-summary pairs into a clean dictionary structure:
{"article": ..., "reference_summary": ...}
"""

import sys
from typing import List, Dict


def load_news_dataset(split: str = "test", sample_size: int = 5) -> List[Dict[str, str]]:
    """
    Load article and reference summary pairs from HuggingFace CNN/DailyMail dataset.

    Args:
        split (str): Dataset split ('test', 'train', 'validation'). Default is 'test'.
        sample_size (int): Number of sample articles to retrieve.

    Returns:
        List[Dict[str, str]]: List of dicts with 'article' and 'reference_summary' keys.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("The 'datasets' library is required. Install it via 'pip install datasets'.")

    print(f"Loading {sample_size} articles from CNN/DailyMail (3.0.0) [{split} split]...")
    
    try:
        # Load dataset slice (using canonical repo path 'abisee/cnn_dailymail')
        dataset_slice = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"{split}[:{sample_size}]")
        data = []
        for item in dataset_slice:
            data.append({
                "article": item.get("article", ""),
                "reference_summary": item.get("highlights", "")
            })
        return data
    except Exception as e:
        print(f"Warning: Could not fetch dataset from HuggingFace ({e}). Using offline sample dataset.")
        fallback_data = [
            {
                "article": "(CNN) -- High winds and heavy rains disrupted power for over 50,000 residents across coastal districts on Monday. Local authorities urged citizens to stay indoors while emergency crews worked around the clock to restore services. Flight operations at the central hub were delayed by up to three hours.",
                "reference_summary": "Severe weather caused widespread power outages for 50,000 residents and delayed flights."
            },
            {
                "article": "(CNN) -- Tech giant Aethel announced a breakthrough quantum computing chip capable of performing error-corrected calculations 100 times faster than current silicon processors. The research team plans to make the hardware available for cloud beta testing next quarter.",
                "reference_summary": "Aethel unveiled a novel quantum processor offering 100x speedup with cloud beta coming next quarter."
            }
        ]
        return fallback_data[:sample_size]



def main():
    """Standalone CLI test for Module 1."""
    print("--- Testing Module 1: Data Collection ---")
    data = load_news_dataset(split="test", sample_size=2)
    
    for idx, item in enumerate(data, 1):
        print(f"\n--- Sample Article {idx} ---")
        print(f"Article (first 300 chars):\n{item['article'][:300]}...")
        print(f"\nReference Summary:\n{item['reference_summary']}")
    
    print("\nModule 1 test complete successfully.")


if __name__ == "__main__":
    main()
