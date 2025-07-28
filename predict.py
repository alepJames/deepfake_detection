from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image
import torch
import sys
import cv2
import numpy as np
import mediapipe as mp
from model import CNNWithLandmarkFusion
from gradcam import GradCAM
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNWithLandmarkFusion().to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

# Grad-CAM targeting last CNN layer
target_layer = model.cnn.layer4
cam_extractor = GradCAM(model, target_layer)

# MTCNN & Mediapipe setup
mtcnn = MTCNN(image_size=224, margin=20, device=device)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Normalization transform
normalize = transforms.Normalize([0.5]*3, [0.5]*3)

# Landmark extractor
def get_landmarks(img):
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = face_mesh.process(img_bgr)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        vector = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return vector.astype(np.float32), results
    return None, None

# 🔥 Region heat % function
def compute_region_heat_percent(cam, region_masks):
    region_percentages = {}
    total_heat = cam.sum()
    for region_name, mask in region_masks.items():
        region_heat = cam[mask].sum()
        percent = (region_heat / total_heat) * 100 if total_heat > 0 else 0
        region_percentages[region_name] = float(percent)
    return region_percentages

# Main prediction function
def predict(image_path, save_annotated=True):
    print(f" Processing: {image_path}")
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)

    if face is None:
        print(" No face detected.")
        return

    landmarks, mp_result = get_landmarks(img)
    if landmarks is None:
        print(" No landmarks detected.")
        return

    face = normalize(face).unsqueeze(0).to(device)
    landmark_tensor = torch.tensor(landmarks).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(face, landmark_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    label = "Real" if pred.item() == 0 else "Fake"
    conf_pct = confidence.item() * 100
    result_str = f"[✔] {os.path.basename(image_path)} → {label} ({conf_pct:.2f}% confidence)"
    print(result_str)

    # Grad-CAM heatmap generation
    cam = cam_extractor(face, landmark_tensor)
    cam_np = cam.squeeze()
    H, W = cam_np.shape

    # 🔥 Region masks (basic rectangles)
    region_masks = {
        'Eyes':     np.zeros((H, W), dtype=bool),
        'Nose':     np.zeros((H, W), dtype=bool),
        'Mouth':    np.zeros((H, W), dtype=bool),
        'Forehead': np.zeros((H, W), dtype=bool),
        'Jaw':      np.zeros((H, W), dtype=bool),
    }
    region_masks['Eyes'][50:90, 60:160] = True
    region_masks['Nose'][90:130, 85:140] = True
    region_masks['Mouth'][130:170, 80:150] = True
    region_masks['Forehead'][10:50, 60:160] = True
    region_masks['Jaw'][170:210, 70:150] = True

    # 🔥 Region-wise attention
    region_percentages = compute_region_heat_percent(cam_np, region_masks)
    print("\n🔥 Region-wise Attention Distribution:")
    for region, pct in region_percentages.items():
        print(f" - {region}: {pct:.2f}%")

    # Compute attention focus %
    focus_score = (cam > 0.5).sum() / cam.size
    focus_pct = float(focus_score * 100)
    if pred.item() == 1:
        print(f" Deepfake Focus: {focus_pct:.2f}% of face had high attention")
    else:
        print(f" Real face focus: {focus_pct:.2f}% of face had high attention")

    # Overlay
    face_img = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
    heatmap_overlay = cv2.addWeighted(face_img, 0.6, heatmap, 0.4, 0)

    # Landmarks
    landmark_img = face_img.copy()
    if mp_result:
        for lm in mp_result.multi_face_landmarks[0].landmark:
            x = int(lm.x * 224)
            y = int(lm.y * 224)
            cv2.circle(landmark_img, (x, y), 1, (255, 255, 0), -1)

    # Visualization output
    fig, axs = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1, 0.05]})
    axs[0].imshow(cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Grad-CAM")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Facial Landmarks")
    axs[1].axis("off")

    norm = Normalize(vmin=0, vmax=1)
    cb = ColorbarBase(axs[2], cmap=cm.jet, norm=norm)
    cb.set_label("Attention Intensity")

    os.makedirs("output", exist_ok=True)
    final_path = os.path.join("output", f"sidebyside_{os.path.basename(image_path)}")
    plt.tight_layout()
    plt.savefig(final_path)
    plt.close()
    print(f" ✅ Saved combined image: {final_path}")

    return result_str

# Run from terminal
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict(sys.argv[1])
