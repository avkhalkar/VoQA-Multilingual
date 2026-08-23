def translate_to_english(translator, sentence, src_lang="auto"):
    """Validates structural back-translation logic."""
    if not sentence: return ""
    if src_lang == 'eng': return sentence
    try:
        text_output, _ = translator.predict(input=sentence[:4096], task_str="t2tt", tgt_lang="eng", src_lang=src_lang)
        return str(text_output[0])
    except Exception: return sentence
