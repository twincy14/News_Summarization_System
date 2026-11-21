"""
Module 5: Main Pipeline

Integrates Modules 1 to 4 into an automated pipeline:
1. Load CNN/DailyMail articles (Module 1)
2. Preprocess article text (Module 2)
3. Generate abstractive summary (Module 3)
4. Evaluate generated summary vs reference using ROUGE (Module 4)
5. Output structured benchmark summary table
"""

import sys
import time
from typing import Optional
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dataset import load_news_dataset
from preprocess import preprocess_article
from summarizer import NewsSummarizer
from evaluate import compute_rouge, aggregate_rouge


def run_pipeline(sample_size: int = 3, model_name: str = "facebook/bart-large-cnn", structured: bool = False):
    """
    Run full news summarization and evaluation pipeline.

    Args:
        sample_size (int): Number of articles to evaluate.
        model_name (str): HuggingFace summarization model name.
        structured (bool): Whether to generate structured executive key takeaways.
    """
    print("=" * 75)
    mode_str = "Structured Executive Takeaways" if structured else "Standard Paragraph"
    print(f"Starting News Summarization Pipeline (Samples: {sample_size}, Mode: {mode_str})")
    print("=" * 75)

    # 1. Load Data
    data = load_news_dataset(split="test", sample_size=sample_size)
    print(f"Loaded {len(data)} test articles.\n")

    # 2. Initialize Model Engine
    summarizer = NewsSummarizer(model_name=model_name)

    results = []
    all_rouge_scores = []

    # 3. Process Articles
    for idx, item in enumerate(data, 1):
        raw_article = item["article"]
        reference = item["reference_summary"]

        print(f"Processing Article [{idx}/{len(data)}]...")
        
        # Module 2: Preprocess
        clean_text = preprocess_article(raw_article, preserve_paragraphs=True)
        
        # Module 3: Summarize
        start_time = time.time()
        if structured:
            generated_summary = summarizer.generate_structured_summary(clean_text)
        else:
            generated_summary = summarizer.generate_summary(clean_text, thorough=True)
        gen_time = time.time() - start_time

        
        # Module 4: Evaluate
        rouge_res = compute_rouge(generated_summary, reference)
        all_rouge_scores.append(rouge_res)

        results.append({
            "Article ID": idx,
            "Article Preview": clean_text[:80] + "...",
            "Reference Summary": reference[:80] + "...",
            "Generated Summary": generated_summary[:80] + "...",
            "ROUGE-1 F1": rouge_res["rouge1"]["fmeasure"],
            "ROUGE-2 F1": rouge_res["rouge2"]["fmeasure"],
            "ROUGE-L F1": rouge_res["rougeL"]["fmeasure"],
            "Time (s)": round(gen_time, 2)
        })

    # 4. Display Results Table
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 75)
    print("PER-ARTICLE EVALUATION SUMMARY")
    print("=" * 75)
    print(df_results.to_string(index=False))

    # 5. Display Average ROUGE Metrics
    avg_rouge = aggregate_rouge(all_rouge_scores)
    print("\n" + "=" * 75)
    print("OVERALL AVERAGE ROUGE BENCHMARK SCORES")
    print("=" * 75)
    for metric, score_dict in avg_rouge.items():
        print(f"  {metric.upper():<10} | Precision: {score_dict['precision']:.4f} | Recall: {score_dict['recall']:.4f} | F1-Score: {score_dict['fmeasure']:.4f}")
    print("=" * 75)

    return df_results, avg_rouge


if __name__ == "__main__":
    run_pipeline(sample_size=3)
