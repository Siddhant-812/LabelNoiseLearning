import torch, random, os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from torchvision import transforms
from transformers import ViTModel
import torchinfo
torch.cuda.manual_seed(42)

class TeacherViT(nn.Module):
    def __init__(self, device: torch.device, num_classes: int):
        super(TeacherViT, self).__init__()
        self.C = num_classes
        self.device = device
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', add_pooling_layer=False).to(self.device)
        self.cls_n = nn.Linear(self.vit.config.hidden_size, self.C).to(self.device)
        self.cls_p = nn.Linear(self.vit.config.hidden_size, self.C).to(self.device)

    def forward(self, x: torch.tensor) -> dict:
        """x.shape = (b, 3, 224, 224)"""
        outputs = self.vit(pixel_values=x) 
        hidden_state = outputs.last_hidden_state[:, 0, :]  # CLS token output (b, d)
        logits_n = self.cls_n(hidden_state) # (b, C)
        logits_p = self.cls_p(hidden_state) # (b, C)
        return {"logits_n": logits_n, "logits_p": logits_p, "hidden_state": hidden_state}

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    model = TeacherViT(device="cuda", num_classes=100)
    print(model)
    torchinfo.summary(model)
    a = torch.rand(size=(4, 3, 224, 224)).to("cuda")
    print(model(a))
    print("DONE")