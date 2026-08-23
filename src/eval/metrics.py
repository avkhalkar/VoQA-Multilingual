import time
from bert_score import score

def compute_semantic_score(cands: list, refs: list, semantic_threshold: float = 0.90):
    """
    Safely wraps HuggingFace BERTScore evaluation with automated retries and explicit binarization.
    Batches evaluation heavily on the GPU natively instead of line-by-line bottlenecks.
    
    Args:
        cands (list): Master array of string predictions outputted by the model natively (e.g. ["a red car", ...])
        refs (list): Parallel array of strict Ground Truth answers scraped from GQA (e.g. ["red", ...])
        semantic_threshold (float): Override float limit converting analog Cosine Similarity into boolean True/False.
        
    Returns: list_of_bool_matches, avg_f1, total_accuracy_percentage, raw_f1_list
    """
    for attempt in range(3):
        try:
            P, R, F1 = score(cands, refs, lang="en", verbose=False)
            avg_f1 = F1.mean().item()
            bert_soft_accuracy = (F1 >= semantic_threshold).float().mean().item() * 100
            
            # Form boolean match list natively matching F1 tensor indices
            is_match = [bool((val >= semantic_threshold).item()) for val in F1]
            return is_match, avg_f1, bert_soft_accuracy, F1.tolist()
        except Exception as e:
            print(f"Network error on BERTScore retry {attempt+1}: {e}")
            time.sleep(2)
            
    # Absolute failure state fallback
    print("Failed to calculate BERTScore after 3 retries. Setting F1 to 0.")
    return [False]*len(cands), 0.0, 0.0, [0.0]*len(cands)
