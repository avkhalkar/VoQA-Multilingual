from pathlib import Path

# Automatically infers the base VoQA-Multilingual root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------
# Global Constants
# -----------------------------------------------------
DATA_MULTILINGUAL_DIR = PROJECT_ROOT / "data_multilingual"
FONT_PATH = PROJECT_ROOT / "src" / "utils" / "DejaVuSans-Bold.ttf"

# -----------------------------------------------------
# Extensible Dataset Registry
# -----------------------------------------------------
DATASETS = {
    "gqa": {
        "images": PROJECT_ROOT / "voqa_gqa" / "test" / "gqa" / "images",
        "gt_questions": PROJECT_ROOT / "voqa_gqa" / "test" / "gqa" / "testdev_balanced_questions.json"
    },
    "pope": {
        # Placeholders for future datasets
        "images": PROJECT_ROOT / "voqa_pope" / "images",
        "gt_questions": PROJECT_ROOT / "voqa_pope" / "testdev_balanced_questions.json"
    }
}

# -----------------------------------------------------
# Extensible Model Registry
# -----------------------------------------------------
MODELS = {
    "internvl_1b": "OpenGVLab/InternVL2_5-1B",
    "deepseek_1b": "deepseek-ai/deepseek-vl2-tiny",
    "qwen_3b": "Qwen/Qwen2.5-VL-3B-Instruct"
}
