from transformers import AutoImageProcessor, CLIPImageProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from . import register_vision_tower
from .base import VisionTower

@register_vision_tower('internvl')      
class InternVLVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # HuggingFace AutoModel strictly checks a static dictionary of configs (like CLIPConfig), 
        # instantly throwing a ValueError when handed an unmapped proprietary InternVisionConfig.
        # We manually bypass the dictionary and invoke the class pointer securely from the physical schema file.
        model_class = get_class_from_dynamic_module("modeling_intern_vit.InternVisionModel", cfg.model_name_or_path, trust_remote_code=True)
        self._vision_tower = model_class(cfg)
        
        try:
            self._image_processor = AutoImageProcessor.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
        except:
            self._image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
