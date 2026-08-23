import sys
import argparse
import json
import torch
import math
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# Enforce secure module importing
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import DATASETS, MODELS, PROJECT_ROOT
from src.model.prompts import PROMPTS_DICT
from src.model.few_shot import build_few_shot_prompt
from src.model.ocr_utils import inject_ocr_metadata

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate standard tile ratios
    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_num and i * j >= min_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(list(target_ratios), key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
        
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def fetch_dataset(dataset_key, lang):
    if lang == "eng":
        d_path = PROJECT_ROOT / "data_multilingual" / "subsets" / "subset_1k.jsonl"
    else:
        d_path = PROJECT_ROOT / "data_multilingual" / "translations" / lang / "questions.jsonl"
        
    with open(d_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gqa", help="Target dataset mapped in src/config.py")
    parser.add_argument("--model", type=str, default="internvl_1b", help="Target architecture mapped in src/config.py")
    parser.add_argument("--lang", type=str, default="eng", help="Target execution language vector")
    parser.add_argument("--shots", type=int, default=0, choices=[0, 1, 2, 4, 8], help="Number of few-shot parameters to prepend sequentially")
    parser.add_argument("--prompt_type", type=str, default="short_workflow", help="Zero-shot prompt mapping keys natively from prompts.py")
    parser.add_argument("--image_dir", type=str, default="auto", help="Physical path to images (e.g. output_images/eng/watermark) or 'auto' for raw")
    parser.add_argument("--use_ocr", action="store_true", help="Activates bounding-box injections exclusively for structural zero-shot targets")
    args = parser.parse_args()

    if args.dataset not in DATASETS:
        raise ValueError(f"Dataset {args.dataset} not found in config.py")
    if args.model not in MODELS:
        raise ValueError(f"Model ID {args.model} mapping failed config check.")
    if args.prompt_type not in PROMPTS_DICT:
        raise ValueError(f"Prompt type {args.prompt_type} not found in PROMPTS_DICT.")

    model_path = MODELS[args.model]
    print(f"Deploying Inference Network:")
    print(f" -> Architecture: {args.model.upper()} ({model_path})")
    print(f" -> Dataset Context: {args.dataset.upper()} ({args.lang.upper()})")
    print(f" -> Workflow: {args.prompt_type} (Few-Shot: {args.shots})")
    print("Loading InternVL Model Weights securely to VRAM...")

    # Load Model natively across HuggingFace API formats using the exact bfloat16 requirements
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    
    generation_config = dict(max_new_tokens=1024, do_sample=True, temperature=0.2)
    
    dataset_records = fetch_dataset(args.dataset, args.lang)
    
    # ---------------------------------------------------------
    # SAFETY FIREWALL: Explicit Pre-Flight Logic Validation
    # ---------------------------------------------------------
    if args.prompt_type == "baseline":
        if args.shots != 0:
            raise ValueError("CRITICAL ERROR: Traditional VQA (baseline) strictly forbids few-shot parameters! Must be 0-shot.")
        images_dir = Path(DATASETS[args.dataset]["images"]) # Force Raw Images
    
    elif args.prompt_type in ["no_prompt", "light"]:
        if args.shots != 0:
            raise ValueError(f"CRITICAL ERROR: Prompt '{args.prompt_type}' structurally forbids few-shot patterns! Must be 0-shot.")
        if args.image_dir == "auto":
            raise ValueError("CRITICAL ERROR: You must explicitly provide an --image_dir (watermark/concat) for rendered evaluations!")
        images_dir = Path(args.image_dir)
        
    elif args.prompt_type in ["short_workflow", "long_workflow"]:
        if args.image_dir == "auto":
            raise ValueError("CRITICAL ERROR: You must explicitly provide an --image_dir (watermark/concat) for rendered evaluations!")
        images_dir = Path(args.image_dir)
        
    if args.use_ocr:
        if args.prompt_type not in ["short_workflow", "long_workflow"]:
            raise ValueError("CRITICAL ERROR: OCR mappings are strictly physically impossible on unstructured prompts! Must be short/long workflow.")
        if args.shots != 0:
            raise ValueError("CRITICAL ERROR: OCR mappings fundamentally corrupt few-shot ablation targets! Must strictly be evaluated as 0-shot.")
    # ---------------------------------------------------------
    
    out_dir = PROJECT_ROOT / "output" / args.model / args.dataset / args.lang
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # DYNAMIC METADATA INJECTION 
    # Prevents overwriting files across Rendering modes or OCR activations
    # ---------------------------------------------------------
    exp_suffix = ""
    if args.use_ocr:
        exp_suffix += "_ocr"
    if args.image_dir != "auto":
        parent_name = str(Path(args.image_dir).parent.name)
        if "watermark" in parent_name: exp_suffix += "_watermark"
        elif "concat_resizing" in parent_name: exp_suffix += "_concat_resize"
        elif "concat_padding" in parent_name: exp_suffix += "_concat_pad"
        
    out_file = out_dir / f"results_{args.prompt_type}_{args.shots}shot{exp_suffix}.jsonl"
    
    # ---------------------------------------------------------
    # RESUME CHECKPOINT: Load already-processed question IDs
    # ---------------------------------------------------------
    processed_ids = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as existing_f:
            for line in existing_f:
                if line.strip():
                    try:
                        processed_ids.add(json.loads(line)["question_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        
        if len(processed_ids) >= len(dataset_records):
            print(f"CHECKPOINT: {out_file.name} already complete ({len(processed_ids)}/{len(dataset_records)}). Skipping.")
            return
        elif len(processed_ids) > 0:
            print(f"CHECKPOINT: Resuming from {len(processed_ids)}/{len(dataset_records)} completed samples.")
    
    print(f"Beginning Inference Loop over {len(dataset_records)} samples ({len(processed_ids)} already done)...")
    with open(out_file, "a", encoding="utf-8") as out_f:
        for idx, item in enumerate(tqdm(dataset_records)):
            # Skip already-processed samples for resume safety
            if item.get("question_id") in processed_ids:
                continue
                
            img_path = images_dir / item.get("image", "")
            if not img_path.exists(): continue
            
            # ---------------------------------------------------------
            # HIGH-READABILITY HARDCODED BRANCH LOGIC
            # ---------------------------------------------------------
            if args.prompt_type == "baseline":
                # Branch 1: Traditional VQA (Strictly Raw Images + Zero Shot + Text Injection)
                base_prompt = PROMPTS_DICT["baseline"]
                final_prompt = base_prompt.replace("{question}", item.get("text", ""))
                final_prompt = "<image>\n" + final_prompt
                
            elif args.prompt_type in ["no_prompt", "light"]:
                # Branch 2: Unstructured Rendering Evaluation (Rendered Images + Strictly Zero Shot)
                base_prompt = PROMPTS_DICT[args.prompt_type]
                final_prompt = "<image>\n" + base_prompt
                
            elif args.prompt_type in ["short_workflow", "long_workflow"]:
                # Branch 3: Strict Structural Workflows (Rendered Images + Zero-shot OR Few-shot)
                context_prompt = ""
                context_pixel_arrays = []
                if args.shots > 0:
                    history_slice = dataset_records[max(0, idx-args.shots):idx]
                    
                    # Physically load the historical memory pixels natively bridging the VLM loop
                    for e in history_slice:
                        hist_path = images_dir / e.get("image", "")
                        if hist_path.exists():
                            context_pixel_arrays.append(load_image(hist_path, max_num=12))
                            
                    examples = [{"question": e.get("text", "question"), "answer": e.get("answer", "...")} for e in history_slice]
                    
                    # Edgecase padding if slice is smaller than required shots
                    while len(examples) < args.shots:
                        examples.append({"question": "Example Context?", "answer": "Synthesized Baseline."})
                        context_pixel_arrays.append(load_image(img_path, max_num=12)) # Duplicate current image to appease token padding
                        
                    context_prompt = build_few_shot_prompt(examples)
                    
                base_prompt = PROMPTS_DICT[args.prompt_type]
                
                # Natively inject physical OCR coordinates if explicitly triggered
                if args.use_ocr:
                    with Image.open(img_path) as tmp_img:
                        img_w, img_h = tmp_img.size
                    bbox_data = item.get("bbox", "Unknown")
                    if isinstance(bbox_data, list) and len(bbox_data) == 4:
                        top_left = f"({bbox_data[0]}, {bbox_data[1]})"
                        bot_right = f"({bbox_data[2]}, {bbox_data[3]})"
                    else:
                        top_left, bot_right = "Unknown", "Unknown"
                    base_prompt = inject_ocr_metadata(base_prompt, img_w, img_h, str(bbox_data), top_left, bot_right)
                    
                final_prompt = context_prompt + base_prompt
            # ---------------------------------------------------------
            
            # Formally replace the <image> macro required intrinsically by InternVL tokenizer formats natively
            if "<image>" not in final_prompt:
                final_prompt = "<image>\n" + final_prompt
            
            try:
                # Load the primary targeted image structure natively into 448x448 absolute tiles
                target_pixel_values = load_image(img_path, max_num=12)
                
                # Merge the historical contexts + target matrices together if strictly evaluated in few-shot
                if args.prompt_type in ["short_workflow", "long_workflow"] and args.shots > 0:
                    all_matrices = context_pixel_arrays + [target_pixel_values]
                    pixel_values = torch.cat(all_matrices, dim=0).to(torch.bfloat16).cuda()
                else:
                    pixel_values = target_pixel_values.to(torch.bfloat16).cuda()
                
                # Dynamic inference pipeline 
                response = model.chat(tokenizer, pixel_values, final_prompt, generation_config)
                
            except Exception as e:
                response = f"Runtime Inference Error: {str(e)}"
                
            # Log output and flush immediately for checkpoint safety
            result_item = {
                "question_id": item.get("question_id"),
                "image": item.get("image"),
                "text": item.get("text"),
                "ground_truth": item.get("answer", ""),
                "model_prediction": response
            }
            out_f.write(json.dumps(result_item) + "\n")
            out_f.flush()
            
    print(f"Inference Completed. Results safely dumped structurally to: {out_file}")

if __name__ == "__main__":
    main()
