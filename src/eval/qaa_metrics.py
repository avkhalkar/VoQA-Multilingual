def calculate_edit_distance(s1: str, s2: str) -> int:
    """Computes dynamic programming Levenshtein edit distance for QAA calculations."""
    if len(s1) < len(s2):
        return calculate_edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]

def calculate_qaa_score(q_hat: str, gt_question: str) -> float:
    """
    Computes Question Alignment Accuracy between Ground Truth and the VLM detection.
    
    Formula citation (VoQA Methodology):
    QAA = 1 - (min(EditDistance(q_hat, q)) / len(q))
    """
    if len(gt_question) == 0 or not q_hat:
        return 0.0
    distance = calculate_edit_distance(q_hat.lower(), gt_question.lower())
    ratio = distance / float(len(gt_question))
    return max(0.0, 1.0 - ratio)
