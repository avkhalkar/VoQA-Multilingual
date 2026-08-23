def build_few_shot_prompt(examples):
    """
    Constructs a Few-Shot prompt dynamically based on the number of examples (1, 2, 4, 8).
    `examples` should be a list of dicts: [{'question': '...', 'answer': '...'}]
    NOTE: The caller must also handle interleaving the actual image tensors in the VLM input!
    """
    prompt = "Each image contains a question written inside it. Your task is to extract the question from the image and answer it accurately.\n"
    for i, ex in enumerate(examples):
        prompt += f"Example {i+1}:\nInput:\n<image>\nOutput:\n{{\"The question in the image\": \"{ex['question']}\", \"Answer\": \"{ex['answer']}\"}}\n"
    prompt += "Now answer the next one:\nInput:\n<image>\nOutput:\n"
    return prompt
