def inject_ocr_metadata(prompt_text, img_w, img_h, bbox_str="Unknown", top_left="Unknown", bot_right="Unknown"):
    """
    Dynamically injects OCR-assisted bounding box context if OCR is enabled.
    """
    return prompt_text.replace("<picture-width>", str(img_w)) \
                      .replace("<picture-height>", str(img_h)) \
                      .replace("<bbox>", bbox_str) \
                      .replace("<top-left-location>", top_left) \
                      .replace("<bottom-right-location>", bot_right)

