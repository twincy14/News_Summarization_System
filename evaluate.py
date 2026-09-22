"""
Module 4: ROUGE Metric Evaluation Engine

Evaluates generated summaries against human reference summaries using:
- ROUGE-1 (Unigram overlap)
- ROUGE-2 (Bigram overlap)
- ROUGE-L (Longest Common Subsequence)
"""

from typing import Dict, List


def compute_rouge(prediction: str, reference: str) -> Dict[str, Dict[str, float]]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L metrics for a generated prediction vs reference.

    Args:
        prediction (str): Model generated summary.
        reference (str): Ground truth human reference summary.

    Returns:
        Dict[str, Dict[str, float]]: ROUGE scores mapping metric name to
                                    {'precision': float, 'recall': float, 'fmeasure': float}
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("The 'rouge-score' library is required. Install it via 'pip install rouge-score'.")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(target=reference, prediction=prediction)

    result = {}
    for metric, score in scores.items():
        result[metric] = {
            "precision": round(score.precision, 4),
            "recall": round(score.recall, 4),
            "fmeasure": round(score.fmeasure, 4)
        }

    return result


def aggregate_rouge(scores_list: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """
    Compute average ROUGE scores across a list of evaluation results.

    Args:
        scores_list (List[Dict]): List of dictionary scores computed per item.

    Returns:
        Dict[str, Dict[str, float]]: Averaged ROUGE metrics.
    """
    if not scores_list:
        return {}

    metrics = ["rouge1", "rouge2", "rougeL"]
    sub_keys = ["precision", "recall", "fmeasure"]

    totals = {m: {k: 0.0 for k in sub_keys} for m in metrics}

    for item_score in scores_list:
        for m in metrics:
            if m in item_score:
                for k in sub_keys:
                    totals[m][k] += item_score[m].get(k, 0.0)

    count = len(scores_list)
    averaged = {}
    for m in metrics:
        averaged[m] = {k: round(totals[m][k] / count, 4) for k in sub_keys}

    return averaged


def main():
    """Standalone CLI test for Module 4."""
    print("--- Testing Module 4: Evaluation ---")
    
    reference = "CERN scientists observed unexpected subatomic particle behavior during LHC collisions."
    generated = "CERN researchers discovered unusual quantum particle interactions in the Large Hadron Collider."
    
    scores = compute_rouge(generated, reference)
    
    print("\nReference:", reference)
    print("Generated:", generated)
    print("\n[ROUGE Scores]:")
    for metric, val in scores.items():
        print(f"  {metric.upper()}: Precision={val['precision']}, Recall={val['recall']}, F1={val['fmeasure']}")
    
    print("\nModule 4 test complete successfully.")


if __name__ == "__main__":
    main()
