import torch
from transformers import CLIPProcessor
from models.teacher import get_teacher_model

def generate_text_anchors(class_names, clip_model_name="openai/clip-vit-base-patch32", device="cuda"):
    """Extracts normalized text embeddings for all dataset class prompts."""
    print("Generating Fixed Text Anchors from Teacher...")
    teacher = get_teacher_model(clip_model_name, device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    
    prompts = [f"a photo of a {c}" for c in class_names]
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        if hasattr(teacher, "get_text_features"):
            txt_feats = teacher.get_text_features(**inputs)
        else:
            txt_out = teacher.text_model(**inputs)
            pooled_output = txt_out.pooler_output if hasattr(txt_out, 'pooler_output') else txt_out[1]
            txt_feats = teacher.text_projection(pooled_output)

        if not isinstance(txt_feats, torch.Tensor):
            txt_feats = txt_feats.pooler_output if hasattr(txt_feats, 'pooler_output') else txt_feats[0]

        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)

    del teacher
    if device == "cuda":
        torch.cuda.empty_cache()
        
    return txt_feats