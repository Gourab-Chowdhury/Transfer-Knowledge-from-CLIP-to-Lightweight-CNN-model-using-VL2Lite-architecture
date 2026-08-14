import torch
import torch.nn as nn
from transformers import CLIPModel

def get_teacher_model(model_name="openai/clip-vit-base-patch32", device="cuda"):
    """Loads and freezes the CLIP Teacher model."""
    print(f"Loading Teacher Model ({model_name})...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def extract_teacher_image_features(teacher_model, images):
    """Safely extracts normalized image features from the CLIP vision encoder."""
    with torch.no_grad():
        if hasattr(teacher_model, "get_image_features"):
            features = teacher_model.get_image_features(images)
        else:
            vision_outputs = teacher_model.vision_model(images)
            pooled_output = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs[1]
            features = teacher_model.visual_projection(pooled_output)

        if not isinstance(features, torch.Tensor):
            features = features.pooler_output if hasattr(features, 'pooler_output') else features[0]

        return features / features.norm(p=2, dim=-1, keepdim=True)