"""
Universal Source of Truth for Zero-Shot and Few-Shot Experimental Prompts.
Centralizes the prompt configurations evaluated perfectly to VoQA Appendix B.
"""

# All standard zero-shot prompt pipelines.
# Note: JSON curly braces are escaped explicitly with {{ and }} due to downstream assumptions,
# however Python .format() on these strings directly should be avoided in favor of `inject_ocr_metadata` str replacements.
PROMPTS_DICT = {
    # 1. Traditional VQA Baseline (Provides the explicit question as text)
    "baseline": "{question}",
    
    # 2. Pure Visual-Only (No prompt instruction at all)
    "no_prompt": "",
    
    # 3. Light Workflow (VoQA Appendix B.1, Table A3)
    "light": "There is a question in this image, you need to find the question and answer the question based on the visual information of the entire image.",
    
    # 4. Short Workflow (VoQA Appendix B.1 JSON Engine)
    "short_workflow": """Task Definition:
You will receive an image with a watermark question.
Your task is to:
1. Detect and extract the full question text.
2. Locate the question bounding box <bbox> (top-left <top-left-location>, bottom-right <bottom-right-location>) in the <picture-width>x<picture-height> image.
3. Understand the question type.
4. Answer accurately based only on the visual content.
Output Format (strict JSON):
{{
"Detected Question": "<recognized question text>",
"Answer": "<concise answer based on the image>",
"Reasoning": "<brief explanation of how the answer was derived>"
}}
Important:
- Ensure full and coherent question extraction.
- Base the answer strictly on visual evidence.
- If uncertain or unclear, output "Unknown".
- Do not add commentary outside the JSON.
Now, analyze the image, detect the question and its location, and output the result in the required JSON format.""",
    
    # 5. Long Workflow (VoQA Appendix B.1 Strict Reasoning Engine)
    "long_workflow": """(1) Task Definition:
You will receive an image containing a watermark-embedded question. Your task is to:
1. Detect the full question text embedded as a watermark in the image.
2. Understand the meaning of the question.
3. Answer the question based solely on the visual content of the image.
4. Provide an accurate and relevant answer.
(2) Workflow Steps:
Step 1: Watermark Question Detection
- Scan the image for textual content, including semi-transparent overlays and repeated patterns.
- Extract the complete question sentence.
- Tip: The question is located at bounding box <bbox>, from top-left <top-left-location> to bottom-right <bottom-right-location> in a <picture-width>x<picture-height> image.
Step 2: Visual Information Extraction
- Analyze the image to find information relevant to the question.
- Focus on object recognition, counting, attribute description, spatial relations, scene understanding, text recognition, or reasoning as needed.
Step 3: Answer Generation
- Provide an accurate, concise answer based solely on visual evidence.
- Give a brief explanation of how the answer was derived.
- If the answer cannot be determined, state it honestly.
(3) Output Format (JSON):
Strictly return a valid JSON object as shown below:
{{
"Detected Question": "<recognized question text>",
"Answer": "<concise answer based on the image>",
"Reasoning": "<brief explanation of how the answer was derived>"
}}
Example:
{{
"Detected Question": "What is the brand of this camera?",
"Answer": "Canon",
"Reasoning": "The text 'Canon' is clearly visible on the camera body."
}}
(4) Important Notes:
1. Ensure question detection and answer are grounded in the image.
2. Provide the full, coherent question text.
3. Base the answer strictly on visual evidence.
4. Be concise and clear.
5. State uncertainty clearly if the image lacks information.
Now, process the input image and execute the VoQA task following the above workflow."""
}
