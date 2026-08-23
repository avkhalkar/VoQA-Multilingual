import json
import re

def parse_vlm_json(raw_text):
    """
    VoQA Appendix B.2 - Pre-Translation Response Filtering Engine
    Safely extracts strict JSON structured predictions dynamically without breaking string boundaries.
    
    Example VLM Hallucination:
        "Sure, here is your answer!
        {
            \"Detected Question\": \"Di che colore è l'auto?\",
            \"Answer\": \"rosso\",
            \"Reasoning\": \"L'auto è rossa.\"
        }
        Hope this helps!"
        
    This parser uses Regex (re.search) to aggressively bypass the conversational wrapper 
    and isolate strictly the bracketed dictionary.
    """
    ans, q_hat = raw_text, ""
    try:
        # Regex aggressively hunts exclusively for the `{ ... }` dictionary payload
        match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            
            # Map robustly across multiple possible output formats depending on shot limits
            ans = str(data.get("final_answer", data.get("Answer", data.get("answer", ans))))
            
            # Specifically extract q_hat (the detected watermark text) used for QAA metrics
            q_hat = str(data.get("Detected Question", data.get("The question in the image", data.get("detected_question", data.get("extracted_question", "")))))
    except Exception: 
        # Fallback organically to raw text if the JSON is violently malformed
        pass
        
    return ans, q_hat
