from rouge_score import rouge_scorer
import dspy


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
        scores["rougeL"].recall,
        scores["rougeL"].precision,
        scores["rougeL"].fmeasure,
    )


def dspy_rouge(example: dspy.Example, prediction: dspy.ChainOfThought):
    precision, recall, fmeasure = rouge_metric(prediction.Clinician, example.Clinician)
    if recall > 0.9 and precision > 0.9:
        return True
    else:
        return False
