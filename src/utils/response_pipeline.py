from src.utils.response_parser import parse_vlm_json
from src.utils.translation_utils import translate_to_english
from src.utils.process_answer import extract_answer

_TRANS_CACHE = {}

def run_response_pipeline(raw_prediction: str, config_name: str, lang: str, translator) -> tuple[str, str, str]:
    """
    Centralized 3-Stage Orchestrator for Response Structuring.
    Dynamically routes text strings through native extraction, translation, and scoring pipelines.
    """
    # ---------------------------------------------------------
    # Stage 1: Pre-Filtering (VOQA Appendix B.2 - Structural JSON Extraction)
    # ---------------------------------------------------------
    # We only spend compute energy parsing JSON if the script was mathematically configured to output JSON
    if "short_workflow" in config_name or "long_workflow" in config_name:
        parsed_pred, q_hat = parse_vlm_json(raw_prediction)
    else:
        parsed_pred = raw_prediction
        q_hat = ""  # No watermark text extraction natively expected
        
    # ---------------------------------------------------------
    # Stage 2: Back-Translation (Language Normalization)
    # ---------------------------------------------------------
    # We strictly bypass the SeamlessM4T loop internally if the subset is already English
    if lang != "eng":
        cache_key = f"{lang}::{parsed_pred}"
        if cache_key in _TRANS_CACHE:
            eng_pred = _TRANS_CACHE[cache_key]
        else:
            eng_pred = translate_to_english(translator, parsed_pred, src_lang=lang)
            _TRANS_CACHE[cache_key] = eng_pred
    else:
        eng_pred = parsed_pred
        
    # ---------------------------------------------------------
    # Stage 3: Post-Filtering (VoQA Appendix B.2 - Post-Translation Heuristics)
    # ---------------------------------------------------------
    # As mandated by VoQA Appendix B.2, we must apply rigorous Exact-Match filters
    # natively after translation. This strips conversational filler ("Yes,"), removes punctuation, 
    # and enforces strict lowercase structural boundaries so "A red car." cleanly becomes "red".
    clean_pred = extract_answer(eng_pred, filter_answer=True, task_name='gqa').lower().strip()
    
    return clean_pred, q_hat, eng_pred
