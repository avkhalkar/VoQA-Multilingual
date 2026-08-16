import json
import re

def extract_heuristic(text):
    text_no_quotes = re.sub(r'"[^"]+"', '', text)
    text_no_quotes = re.sub(r"'[^']+'", '', text_no_quotes)
    text_no_quotes = text_no_quotes.strip().lower()
    words = re.findall(r'\b[a-z0-9]+\b', text_no_quotes)
    if words:
        if words[-1] in ["phrase", "word", "question", "single", "answer"]:
            # If the last words are part of "answer the question using..."
            pass 
        return words[-1]
    return ""

def main():
    questions = json.load(open('voqa_gqa/test/gqa/testdev_balanced_test100_questions.json'))
    predictions = []
    with open('voqa_gqa/test/gqa/answers/llava_gqa_testdev_balanced_test100_prompt0/InternVL2_5-1B_no/gqa_watermark_rendering_image/1_0.jsonl') as f:
        for line in f:
            predictions.append(json.loads(line))

    for p in predictions:
        qid = p['question_id']
        text = p['text'].lower()
        gold = str(questions[qid]['answer']).lower()
        
        words = re.findall(r'\b[a-z0-9]+\b', text)
        if gold in words:
            print(f"GOLD: {gold}")
            print(f"TEXT: {p['text']}")
            print("-" * 40)

if __name__ == "__main__":
    main()
