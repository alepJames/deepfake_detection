import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from model import CNNWithLandmarkFusion
import os
import sys

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNWithLandmarkFusion()
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval().to(device)

# Hook feature maps
feature_maps = {}

def hook_layer(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach().cpu()
    return hook

# Register hooks
model.cnn.conv1.register_forward_hook(hook_layer("conv1"))
model.cnn.layer1.register_forward_hook(hook_layer("layer1"))
model.cnn.layer2.register_forward_hook(hook_layer("layer2"))

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def visualize_feature_maps(image_path):
    if not os.path.exists(image_path):
        print(f"[❌] File not found: {image_path}")
        return

    print(f"[📷] Loading: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    dummy_landmarks = torch.zeros((1, 1404)).to(device)

    _ = model(img_tensor, dummy_landmarks)

    os.makedirs("output", exist_ok=True)

    for name, fmap in feature_maps.items():
        print(f"[🔍] Visualizing {name} → shape: {fmap.shape}")
        fmap = fmap.squeeze(0)

        n = min(8, fmap.shape[0])
        fig, axs = plt.subplots(1, n, figsize=(15, 4))
        for i in range(n):
            axs[i].imshow(fmap[i], cmap='viridis')
            axs[i].axis('off')
            axs[i].set_title(f"{name}[{i}]", fontsize=8)
        plt.tight_layout()
        out_path = f"output/featuremap_{name}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[💾] Saved: {out_path}")

# CLI usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python featuremap_vis.py <image_path>")
    else:
        visualize_feature_maps(sys.argv[1])
