from rouge_score import rouge_scorer


def rouge_metric(predictions, ground_truth):
    """
    Calculate ROUGE score for a list of predictions and ground truth texts.

    Args:
        predictions (list): List of predicted texts.
        ground_truth (list): List of ground truth texts.

    Returns:
        dict: Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    # Initialize the Rouge scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    # Calculate the scores
    scores = scorer.score(ground_truth, predictions)

    return (
        scores["rougeL"].precision,
        scores["rougeL"].recall,
        scores["rougeL"].fmeasure,
    )
