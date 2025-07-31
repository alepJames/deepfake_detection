import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from captum.attr import IntegratedGradients
import os
from model import CNNWithLandmarkFusion

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNWithLandmarkFusion()
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Visualization function
def visualize_attribution(attr, original_image, save_path):
    attr = attr.squeeze().cpu().detach().numpy()
    attr = np.sum(np.abs(attr), axis=0)  # Sum over RGB
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(attr * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Integrated Gradients Attribution")
    plt.axis("off")
    os.makedirs("output", exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[✔] Saved IG visualization to: {save_path}")

# Run IG attribution
def run_integrated_gradients(image_path):
    print(f"\n[INFO] Running Integrated Gradients for: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    baseline = torch.zeros_like(img_tensor).to(device)

    # Fix: make landmark batch same as image batch
    def model_forward(x):
        B = x.shape[0]
        landmark_batch = torch.zeros((B, 1404)).to(device)
        return model(x, landmark_batch)

    # Run Captum IG
    ig = IntegratedGradients(model_forward)
    attr = ig.attribute(img_tensor, baseline, target=1, n_steps=50)

    # Convert original image for visualization
    original_cv2 = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
    out_path = os.path.join("output", f"ig_{os.path.basename(image_path)}")
    visualize_attribution(attr, original_cv2, out_path)

# CLI entry
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python captum_ig.py <image_path>")
    else:
        run_integrated_gradients(sys.argv[1])
