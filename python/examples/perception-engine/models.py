import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

class CLIPEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        super().__init__()
        self.device = device
        print(f"[init] Chargement de {model_name} sur {device} ...")
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        # CLIP-specific ImageNet-style normalization constants on GPU
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode_images(self, frames_gpu):
        """
        Preprocesses and encodes a batch of frames directly on the GPU.
        Input:
            frames_gpu: PyTorch tensor on CUDA of shape (B, 3, H, W) in [0, 1] range.
        Output:
            Normalized image embeddings: PyTorch tensor on CUDA of shape (B, 512).
        """
        # 1. Resize to 224x224 directly in VRAM
        resized = torch.nn.functional.interpolate(
            frames_gpu, size=(224, 224), mode="bilinear", align_corners=False
        )
        # 2. Normalize in VRAM
        normalized = (resized - self.mean) / self.std
        
        # 3. Run CLIP vision transformer
        image_features = self.model.get_image_features(pixel_values=normalized)
        
        # 4. L2 Normalize for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    @torch.no_grad()
    def encode_text(self, text_list):
        """
        Encodes a list of text queries directly on the GPU.
        Input:
            text_list: list of strings
        Output:
            Normalized text embeddings: PyTorch tensor on CUDA of shape (T, 512).
        """
        inputs = self.tokenizer(text_list, padding=True, return_tensors="pt").to(self.device)
        text_features = self.model.get_text_features(**inputs)
        
        # L2 Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
